#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <gvins/LocalSensorExternalTrigger.h>
#include <sensor_msgs/NavSatFix.h>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

using namespace gnss_comm;

#define MAX_GNSS_CAMERA_DELAY 0.05

std::unique_ptr<Estimator> estimator_ptr;

std::condition_variable con;//条件变量
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<std::vector<ObsPtr>> gnss_meas_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;
//互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;
//IMU项[P,Q,B,Ba,Bg,a,g]
double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = -1;

std::mutex m_time;
double next_pulse_time;
bool next_pulse_time_valid;
double time_diff_gnss_local;
bool time_diff_valid;
double latest_gnss_time;
double tmp_last_feature_time;
uint64_t feature_msg_counter;
int skip_parameter;
//从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    //init_imu=1表示第一个IMU数据
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;//两帧数据时间差
    latest_time = t;
  //得到加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};
    //上一时刻世界坐标系下的加速度值 tmp_Q相当于R_wi
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator_ptr->g;
    //中值陀螺仪的结果
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    //更新姿态。将一个旋转向量，更新到四元数上 
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}
//从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
//对imu_buf中剩余的imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator_ptr->Ps[WINDOW_SIZE];
    tmp_Q = estimator_ptr->Rs[WINDOW_SIZE];
    tmp_V = estimator_ptr->Vs[WINDOW_SIZE];
    tmp_Ba = estimator_ptr->Bas[WINDOW_SIZE];
    tmp_Bg = estimator_ptr->Bgs[WINDOW_SIZE];
    acc_0 = estimator_ptr->acc_0;
    gyr_0 = estimator_ptr->gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}
/**
 * @brief   对imu和图像数据以及GNSS数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -  
 *                  直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据           
 * @return  bool
*/
bool
getMeasurements(std::vector<sensor_msgs::ImuConstPtr> &imu_msg, sensor_msgs::PointCloudConstPtr &img_msg, std::vector<ObsPtr> &gnss_msg)
{
    //如果feature中的数据或者imu中的数据或者GNSS中的数据为空的话，那么直接返回False
    if (imu_buf.empty() || feature_buf.empty() || (GNSS_ENABLE && gnss_meas_buf.empty()))
        return false;
    
    double front_feature_ts = feature_buf.front()->header.stamp.toSec();
        //imu     ******
        //image           ****
        //这就是imu的数据还没有来，这里判断的是：如果最后一个的数据的时间小于第一帧图像的时间，则执行 
    if (!(imu_buf.back()->header.stamp.toSec() > front_feature_ts))
    {
        //ROS_WARN("wait for imu, only should happen at the beginning");
        sum_of_wait++;
        return false;
    }
    //imu      *****
    //image  *******
    //这种只能仍掉一些image帧
    //对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
    double front_imu_ts = imu_buf.front()->header.stamp.toSec();
    while (!feature_buf.empty() && front_imu_ts > front_feature_ts)
    {
        ROS_WARN("throw img, only should happen at the beginning");
        feature_buf.pop();
        front_feature_ts = feature_buf.front()->header.stamp.toSec();
    }
    //允许使用GNSS数据时才会执行
    if (GNSS_ENABLE)
    {   //特征点的时间戳加上GNSS与图像的延迟时间，
        front_feature_ts += time_diff_gnss_local;
        //front_gnss_ts是第一帧GNss数据的时间戳
        double front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        //如果gnss_meas_buf中的数据没有被清空，并且第一帧GNSS的数据小于特征点的时间戳-0.05，那么就会被弹出
        while (!gnss_meas_buf.empty() && front_gnss_ts < front_feature_ts-MAX_GNSS_CAMERA_DELAY)
        {
            ROS_WARN("throw gnss, only should happen at the beginning");
            gnss_meas_buf.pop();
            if (gnss_meas_buf.empty()) return false;
            front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        }
        if (gnss_meas_buf.empty())//多余的判断
        {
            ROS_WARN("wait for gnss...");
            return false;
        }//如果GNSS的数据不在特征点数据的两边，那么就会被弹出
        else if (abs(front_gnss_ts-front_feature_ts) < MAX_GNSS_CAMERA_DELAY)
        {
            gnss_msg = gnss_meas_buf.front();
            gnss_meas_buf.pop();
        }
    }

    img_msg = feature_buf.front();
    feature_buf.pop();

    while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator_ptr->td)
    {//emplace_back相比push_back能更好地避免内存的拷贝与移动
        imu_msg.emplace_back(imu_buf.front());
        imu_buf.pop();
    }
    //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
    //保留图象时间戳后一个imu数据，但是并不会从buffer中扔掉
    //imu  ***   *
    //image    *
    imu_msg.emplace_back(imu_buf.front());
    if (imu_msg.empty())
        ROS_WARN("no imu between two image");
    return true;
}
//imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]，同时按照imu的频率发送。 
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //判断时间间隔是否为正确的
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);//有可能我在存储数据的时候，有线程想要往外取数据，所以需要线程锁
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();//唤醒作用于process线程中的获取观测值数据的函数

    {
        //构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);//递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        //发布最新的由IMU直接递推得到的PQV
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);//转换为一个格式之后，发布出去
    }
}
//订阅gnss星历信息
void gnss_ephem_callback(const GnssEphemMsgConstPtr &ephem_msg)
{
    EphemPtr ephem = msg2ephem(ephem_msg);
    estimator_ptr->inputEphem(ephem);
}
//为什么订阅了两个：因为在两个导航系下，星历的格式是不一样的
//订阅格洛纳斯-星历信息
void gnss_glo_ephem_callback(const GnssGloEphemMsgConstPtr &glo_ephem_msg)
{
    GloEphemPtr glo_ephem = msg2glo_ephem(glo_ephem_msg);
    estimator_ptr->inputEphem(glo_ephem);
}
//电离层参数订阅
//卫星信号在传播的过程中会受到电离层和对流层的影响，且如果建模不正确或不考虑两者的影响，会导致定位结果变差，因此，通常都会对两者进行建模处理；
//后面我们在选择卫星信号时，会考虑卫星的仰角，也是因为对于仰角小的卫星，其信号在电离层和对流层中经过的时间较长，对定位影响大，这样的卫星我们就会排除；
void gnss_iono_params_callback(const StampedFloat64ArrayConstPtr &iono_msg)
{
    double ts = iono_msg->header.stamp.toSec();
    std::vector<double> iono_params;
    std::copy(iono_msg->data.begin(), iono_msg->data.end(), std::back_inserter(iono_params));
    assert(iono_params.size() == 8);
    estimator_ptr->inputIonoParams(ts, iono_params);
}
//订阅GNSS的测量信息
void gnss_meas_callback(const GnssMeasMsgConstPtr &meas_msg)
{
    //将ros消息格式转换为GNSS的测量信息
    std::vector<ObsPtr> gnss_meas = msg2meas(meas_msg);
    //存储上一次的时间
    latest_gnss_time = time2sec(gnss_meas[0]->time);

    // cerr << "gnss ts is " << std::setprecision(20) << time2sec(gnss_meas[0]->time) << endl;
    if (!time_diff_valid)   return;// 如果GNSS与vio的时间是没有对齐的，则退出

    m_buf.lock();
    gnss_meas_buf.push(std::move(gnss_meas));
    m_buf.unlock();
    con.notify_one();
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    ++ feature_msg_counter;
    //skip_parameter在初始的时候，值为-1，所以直接会执行
    if (skip_parameter < 0 && time_diff_valid)
    {   //将feature的时间对齐到GNSS的时间戳
        //
        const double this_feature_ts = feature_msg->header.stamp.toSec()+time_diff_gnss_local;
        if (latest_gnss_time > 0 && tmp_last_feature_time > 0)//证明GNSS还有feature已经开始传入数据了
        {
            if (abs(this_feature_ts - latest_gnss_time) > abs(tmp_last_feature_time - latest_gnss_time))
                skip_parameter = feature_msg_counter%2;       // skip this frame and afterwards
            else
                skip_parameter = 1 - (feature_msg_counter%2);   // skip next frame and afterwards
        }
        // cerr << "feature counter is " << feature_msg_counter << ", skip parameter is " << int(skip_parameter) << endl;
        tmp_last_feature_time = this_feature_ts;
    }

    if (skip_parameter >= 0 && int(feature_msg_counter%2) != skip_parameter)
    {
        m_buf.lock();
        feature_buf.push(feature_msg);
        m_buf.unlock();
        con.notify_one();
    }
}
//获得local和gnss的时间差； publish when VI-Sensor is trigger；
//trigger_msg 记录的是相机被GNSS脉冲触发时的时间，也可以理解成图像的命名（以时间 命名），
//和真正的gnss时间是有区别的，因为存在硬件延迟等，这也是后面为什么校正 local和world时间的原因；
void local_trigger_info_callback(const gvins::LocalSensorExternalTriggerConstPtr &trigger_msg)
{
    std::lock_guard<std::mutex> lg(m_time);

    if (next_pulse_time_valid)
    {
        time_diff_gnss_local = next_pulse_time - trigger_msg->header.stamp.toSec();
        estimator_ptr->inputGNSSTimeDiff(time_diff_gnss_local);
        if (!time_diff_valid)       // just get calibrated
            std::cout << "time difference between GNSS and VI-Sensor got calibrated: "
                << std::setprecision(15) << time_diff_gnss_local << " s\n";
        time_diff_valid = true;
    }
}

void gnss_tp_info_callback(const GnssTimePulseInfoMsgConstPtr &tp_msg)
{
    gtime_t tp_time = gpst2time(tp_msg->time.week, tp_msg->time.tow);
    if (tp_msg->utc_based || tp_msg->time_sys == SYS_GLO)
        tp_time = utc2gpst(tp_time);
    else if (tp_msg->time_sys == SYS_GAL)
        tp_time = gst2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_BDS)
        tp_time = bdt2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_NONE)
    {
        std::cerr << "Unknown time system in GNSSTimePulseInfoMsg.\n";
        return;
    }
    double gnss_ts = time2sec(tp_time);

    std::lock_guard<std::mutex> lg(m_time);
    next_pulse_time = gnss_ts;
    next_pulse_time_valid = true;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator_ptr->clearState();
        estimator_ptr->setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;//并没有被用到
        std::vector<sensor_msgs::ImuConstPtr> imu_msg;
        sensor_msgs::PointCloudConstPtr img_msg;
        std::vector<ObsPtr> gnss_msg;
        //获得三者的观测信息，为后面的融合做准备
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
                    return getMeasurements(imu_msg, img_msg, gnss_msg);
                 });
        lk.unlock();
        m_estimator.lock();
        //IMU数据处理
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        for (auto &imu_data : imu_msg)
        {
            double t = imu_data->header.stamp.toSec();
            double img_t = img_msg->header.stamp.toSec() + estimator_ptr->td;
            if (t <= img_t)
            { 
                if (current_time < 0)
                    current_time = t;
                double dt = t - current_time;
                ROS_ASSERT(dt >= 0);
                current_time = t;//保IMU证数据一帧帧都是连续
                dx = imu_data->linear_acceleration.x;
                dy = imu_data->linear_acceleration.y;
                dz = imu_data->linear_acceleration.z;
                rx = imu_data->angular_velocity.x;
                ry = imu_data->angular_velocity.y;
                rz = imu_data->angular_velocity.z;
                //进行IMU预积分
                estimator_ptr->processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

            }
            else//对最后一帧imu数据，做一个插值
            {
                double dt_1 = img_t - current_time;
                double dt_2 = t - img_t;
                current_time = img_t;
                ROS_ASSERT(dt_1 >= 0);
                ROS_ASSERT(dt_2 >= 0);
                ROS_ASSERT(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_data->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_data->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_data->linear_acceleration.z;
                rx = w1 * rx + w2 * imu_data->angular_velocity.x;
                ry = w1 * ry + w2 * imu_data->angular_velocity.y;
                rz = w1 * rz + w2 * imu_data->angular_velocity.z;
                estimator_ptr->processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
            }
        }

        if (GNSS_ENABLE && !gnss_msg.empty())
            estimator_ptr->processGNSS(gnss_msg);

        ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

        TicToc t_s;
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
        for (unsigned int i = 0; i < img_msg->points.size(); i++)
        {
            int v = img_msg->channels[0].values[i] + 0.5;
            int feature_id = v / NUM_OF_CAM;
            int camera_id = v % NUM_OF_CAM;
            double x = img_msg->points[i].x;
            double y = img_msg->points[i].y;
            double z = img_msg->points[i].z;
            double p_u = img_msg->channels[1].values[i];
            double p_v = img_msg->channels[2].values[i];
            double velocity_x = img_msg->channels[3].values[i];
            double velocity_y = img_msg->channels[4].values[i];
            ROS_ASSERT(z == 1);
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
        estimator_ptr->processImage(image, img_msg->header);

        double whole_t = t_s.toc();
        printStatistics(*estimator_ptr, whole_t);
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world";

        pubOdometry(*estimator_ptr, header);
        pubKeyPoses(*estimator_ptr, header);
        pubCameraPose(*estimator_ptr, header);
        pubPointCloud(*estimator_ptr, header);
        pubTF(*estimator_ptr, header);
        pubKeyframe(*estimator_ptr);
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gvins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator_ptr.reset(new Estimator());
    estimator_ptr->setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    registerPub(n);

    next_pulse_time_valid = false;
    time_diff_valid = false;
    latest_gnss_time = -1;
    tmp_last_feature_time = -1;
    feature_msg_counter = 0;

    if (GNSS_ENABLE)
        skip_parameter = -1;
    else
        skip_parameter = 0;
    //订阅IMU信息
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    //订阅前端图像特征点信息
    ros::Subscriber sub_feature = n.subscribe("/gvins_feature_tracker/feature", 2000, feature_callback);
    //订阅故障信息
    ros::Subscriber sub_restart = n.subscribe("/gvins_feature_tracker/restart", 2000, restart_callback);
    //订阅gnss信息
    ros::Subscriber sub_ephem, sub_glo_ephem, sub_gnss_meas, sub_gnss_iono_params;
    ros::Subscriber sub_gnss_time_pluse_info, sub_local_trigger_info;
    if (GNSS_ENABLE)
    {   //订阅星历信息：卫星的位置、速度、时间偏差等信息
        sub_ephem = n.subscribe(GNSS_EPHEM_TOPIC, 100, gnss_ephem_callback);
        sub_glo_ephem = n.subscribe(GNSS_GLO_EPHEM_TOPIC, 100, gnss_glo_ephem_callback);
        //卫星的观测信息
        sub_gnss_meas = n.subscribe(GNSS_MEAS_TOPIC, 100, gnss_meas_callback);
        sub_gnss_iono_params = n.subscribe(GNSS_IONO_PARAMS_TOPIC, 100, gnss_iono_params_callback);
        //GNSS与vio的时间是否同步判断：（因为两者是不同空间的产物，也有可能是不同时间的，因此两者的时间需要进行补偿也是正常的）
        if (GNSS_LOCAL_ONLINE_SYNC)//在线同步，yaml文件中为1，所以首先估计出延迟时间，但是项目中公司有给的时间，
        {
            sub_gnss_time_pluse_info = n.subscribe(GNSS_TP_INFO_TOPIC, 100, 
                gnss_tp_info_callback);
            sub_local_trigger_info = n.subscribe(LOCAL_TRIGGER_INFO_TOPIC, 100, 
                local_trigger_info_callback);
        }
        else
        {
            time_diff_gnss_local = GNSS_LOCAL_TIME_DIFF;//通过yaml文件，直接将延时传递给系统
            estimator_ptr->inputGNSSTimeDiff(time_diff_gnss_local);
            time_diff_valid = true;
        }
    }

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
