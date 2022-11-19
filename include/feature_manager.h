#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"

//某个id的路标点在某帧中的"观测信息"，包括三维坐标（应该是归一化坐标），像素坐标和像素速度
class FeaturePerFrame
{
public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
  {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
  }
  double cur_td;
  Vector3d point;
  Vector2d uv;
  Vector2d velocity;
  double z;
  bool is_used;
  double parallax;
  MatrixXd A;
  VectorXd b;
  double dep_gradient;
};

//某个id的点在它被观察到的所有帧的信息，包括起始帧
class FeaturePerId
{
public:
  const int feature_id;
  int start_frame;
  //某个id的点在被观测到的所有帧中的观测信息
  vector<FeaturePerFrame> feature_per_frame;

  int used_num;
  bool is_outlier;
  bool is_margin;
  double estimated_depth;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  Vector3d gt_p;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
  }

  int endFrame();
};

//这个类包含了所有id的特征点在（滑窗内）所有其被观察到的帧中的信息
class FeatureManager
{
public:
    //参数为Matrix3d类型的数组
  FeatureManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();

  bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  void debugShow();
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  //void updateDepth(const VectorXd &x);
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth(const VectorXd &x);
  VectorXd getDepthVector();
  void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier();
  list<FeaturePerId> feature;
  int last_track_num;

private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  //Rs这个矩阵指的是什么呢？指的是各帧相机坐标系到世界坐标系的旋转
  const Matrix3d *Rs;
  // ric指的是imu到相机的旋转矩阵
  Matrix3d ric[NUM_OF_CAM];
};

#endif