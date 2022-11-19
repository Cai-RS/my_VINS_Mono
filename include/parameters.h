#pragma once

// #include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
#include <fstream>

//feature tracker
// extern int ROW;
// extern int COL;
//NUM_OF_CAM这个常量已经定义了。const对象被设定为仅在文件内有效。当多个文件中出现同名的const变量时，其实等同于在不同文件中分别定义了独立的变量，不会重复定义！
//当然常量也是可以在前面加extern的，这样就可以跟一般变量一样在另一个文件中定义，然后被其他文件声明后使用
const int NUM_OF_CAM = 1;
// extern表示这个全局变量具有外部链接属性，此处仅为声明，可以在其他文件进行定义
extern int FOCAL_LENGTH;
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
// extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern bool STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

//estimator

// const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
// const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

// void readParameters(ros::NodeHandle &n);

void readParameters(std::string config_file);

// 定义了三种新的枚举类型，但都没有声明相应的枚举变量
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
