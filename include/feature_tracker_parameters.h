#pragma once
// #include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

//凡是没有带extern的变量声明同时也都是定义（指申请存储空间）， 如 “int a;”就是即声明又定义
//extern使得变量的声明和定义是可以分开的，这里只进行了变量的声明
//如果想在别的文件中定义或使用这里的全局变量，需要先声明它们，即include这个头文件。
extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;

//NUM_OF_CAM这个常量已经定义了。const对象被设定为仅在文件内有效。当多个文件中出现同名的const变量时，其实等同于在不同文件中分别定义了独立的变量，不会重复定义！
const int NUM_OF_CAM = 1;

extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

void readParameters();
