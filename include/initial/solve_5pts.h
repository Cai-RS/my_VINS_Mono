#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>

class MotionEstimator
{
public:
    //给出一系列的归一化平面上的匹配点（三维，z为1），通过极线约束求解相对位姿
  bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);

private:
    //下面这两个类中的函数只声明了，但是并没有被定义（在整个工程中），使用的是另外定义的cv命名空间中的同功能函数
  double testTriangulation(const vector<cv::Point2f> &l,
                           const vector<cv::Point2f> &r,
                           cv::Mat_<double> R, cv::Mat_<double> t);
  void decomposeE(cv::Mat E,
                  cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                  cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};
