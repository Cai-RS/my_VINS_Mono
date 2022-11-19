#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
	//这里的state表示的是什么？还是说它需要参与sfm，还未被三角化？
    bool state;
    int id;
	//某个路标点的所有被观测值（像素坐标）
    vector<pair<int,Vector2d>> observation;
	// 这个position应该是表示点在第一帧坐标系下的位置，或者是在世界坐标系下的位置
    double position[3];
    double depth;
};

//此部分为代价函数的构造 https://blog.csdn.net/hltt3838/article/details/109694821
struct ReprojectionError3D
{
	//用构造函数传入已知参数（观测值，归一化坐标的前两维）
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	//()重载函数
	//这里三个待估计变量camera_R，camera_T和point均用const修饰，即传入的是常量数组的指针常量，原因是这里仅为了计算残差，避免这里的误操作导致待估计变量的改变（它们只能在slove时被改变）
	// https://blog.csdn.net/weixin_44401286/article/details/118515147
	//残差时如何自动求导的？这里根据残差的计算过程，可以求出残差关于参数中的待估计变量的导数的解析形式？每次代入新传进来的参数即可求得雅可比（对于太复杂的残差计算，是不是就无法自动求导？）
	//有时，无法定义自动求导的模板仿函数，比如参数的估计调用了无法控制的库函数或外部函数,则可以使用数值求导（仍然是系统的自动求导？）
	//当雅可比的计算十分简单，或者十分复杂但又要追求solve时的极致性能，那么可以选择解析求导的方式，但要自己定义雅可比的计算方式
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		//系统是如何在构建大H矩阵时，知道这里的残差是跟哪个待估计量相关？即这个jocobian应该排在H矩阵的哪里？如何寻址？
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		//计算出的归一化坐标的前两维
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
		//注意，还需要将观测值的类型变成跟待估计参数的类型相同
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	//下面为工厂模式函数，为静态函数，意味着在类外部可以不用创建对象就调用此函数，只需在前面加 类名::
	// static ceres::CostFunction*为Create函数的返回类型
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
		// ceres中的求自动求微分中需要给定 代价函数 残差的维度（这里是2），参数块的维度（这里分别是四元数，位移，和点的位置）。待优化变量的传入方式应和 Probelm::AddResidualBlock()一致
		//AutoDiffCostFunction为模板类，是ceres中推荐的课自动求导的代价函数，其构造函数为
		//ceres::AutoDiffCostFunction<CostFunctor, int residualDim, int paramDim>(CostFunctor* functor);
		//其中CostFunctor为仿函数，本质为struct或者class，用于传递已知参数和定义残差的计算过程。这里的仿函数为ReprojectionError3D结构体
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};