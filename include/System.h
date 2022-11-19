#pragma once

#include <stdio.h>
#include <queue>
#include <map>
// https://zhuanlan.zhihu.com/p/194198073
#include <thread>
#include <mutex>

#include <fstream>
//std::condition_variable 和 std::condition_variable_any 是标准库线程同步以条件变量方式的实现
//它的作用是根据设定的条件同步一个或多个线程  https://zhuanlan.zhihu.com/p/484434570
#include <condition_variable>

// #include <cv.h>
// #include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include "estimator.h"
#include "parameters.h"
#include "feature_tracker.h"


//imu for vio
struct IMU_MSG
{
    double header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

//image for vio    
struct IMG_MSG {
    double header;
    //某图像中各特征点的归一化坐标
    vector<Vector3d> points;
    //某帧图像中各特征点的id（从系统启动开始追踪过的所有点）
    vector<int> id_of_point;
    //某帧图像中各特征点的像素信息
    vector<float> u_of_point;
    vector<float> v_of_point;
    vector<float> velocity_x_of_point;
    vector<float> velocity_y_of_point;
};
typedef std::shared_ptr <IMG_MSG const > ImgConstPtr;
    
class System
{
public:
    System(std::string sConfig_files);

    ~System();

    void PubImageData(double dStampSec, cv::Mat &img);

    void PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
        const Eigen::Vector3d &vAcc);

    // thread: visual-inertial odometry
    void ProcessBackEnd();
    void Draw();
    
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;

#ifdef __APPLE__
    void InitDrawGL(); 
    void DrawGLFrame();
#endif

private:

    //feature tracker
    std::vector<uchar> r_status;
    std::vector<float> r_err;
    // std::queue<ImageConstPtr> img_buf;

    // ros::Publisher pub_img, pub_match;
    // ros::Publisher pub_restart;

    FeatureTracker trackerData[NUM_OF_CAM];
    double first_image_time;
    int pub_count = 1;
    bool first_image_flag = true;
    double last_image_time = 0;
    bool init_pub = 0;

    //estimator
    Estimator estimator;

    std::condition_variable con;
    double current_time = -1;
    //这两个队列是用来存储信息，相同时间点只能存或取一个imu和image的信息，通过m_buf来保证这种同步性
    std::queue<ImuConstPtr> imu_buf;
    std::queue<ImgConstPtr> feature_buf;
    // std::queue<PointCloudConstPtr> relo_buf;
    int sum_of_wait = 0;

    std::mutex m_buf;
    //这个锁没被用到
    std::mutex m_state;
    //这个锁没被用到
    std::mutex i_buf;
    std::mutex m_estimator;

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
    double last_imu_t = 0;
    std::ofstream ofs_pose;
    std::vector<Eigen::Vector3d> vPath_to_draw;
    bool bStart_backend;
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
    
};