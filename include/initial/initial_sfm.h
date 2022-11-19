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
	//�����state��ʾ����ʲô������˵����Ҫ����sfm����δ�����ǻ���
    bool state;
    int id;
	//ĳ��·�������б��۲�ֵ���������꣩
    vector<pair<int,Vector2d>> observation;
	// ���positionӦ���Ǳ�ʾ���ڵ�һ֡����ϵ�µ�λ�ã�����������������ϵ�µ�λ��
    double position[3];
    double depth;
};

//�˲���Ϊ���ۺ����Ĺ��� https://blog.csdn.net/hltt3838/article/details/109694821
struct ReprojectionError3D
{
	//�ù��캯��������֪�������۲�ֵ����һ�������ǰ��ά��
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	//()���غ���
	//�������������Ʊ���camera_R��camera_T��point����const���Σ���������ǳ��������ָ�볣����ԭ���������Ϊ�˼���в�����������������´����Ʊ����ĸı䣨����ֻ����sloveʱ���ı䣩
	// https://blog.csdn.net/weixin_44401286/article/details/118515147
	//�в�ʱ����Զ��󵼵ģ�������ݲв�ļ�����̣���������в���ڲ����еĴ����Ʊ����ĵ����Ľ�����ʽ��ÿ�δ����´������Ĳ�����������ſɱȣ�����̫���ӵĲв���㣬�ǲ��Ǿ��޷��Զ��󵼣���
	//��ʱ���޷������Զ��󵼵�ģ��º�������������Ĺ��Ƶ������޷����ƵĿ⺯�����ⲿ����,�����ʹ����ֵ�󵼣���Ȼ��ϵͳ���Զ��󵼣���
	//���ſɱȵļ���ʮ�ּ򵥣�����ʮ�ָ��ӵ���Ҫ׷��solveʱ�ļ������ܣ���ô����ѡ������󵼵ķ�ʽ����Ҫ�Լ������ſɱȵļ��㷽ʽ
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		//ϵͳ������ڹ�����H����ʱ��֪������Ĳв��Ǹ��ĸ�����������أ������jocobianӦ������H�����������Ѱַ��
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		//������Ĺ�һ�������ǰ��ά
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
		//ע�⣬����Ҫ���۲�ֵ�����ͱ�ɸ������Ʋ�����������ͬ
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	//����Ϊ����ģʽ������Ϊ��̬��������ζ�������ⲿ���Բ��ô�������͵��ô˺�����ֻ����ǰ��� ����::
	// static ceres::CostFunction*ΪCreate�����ķ�������
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
		// ceres�е����Զ���΢������Ҫ���� ���ۺ��� �в��ά�ȣ�������2�����������ά�ȣ�����ֱ�����Ԫ����λ�ƣ��͵��λ�ã������Ż������Ĵ��뷽ʽӦ�� Probelm::AddResidualBlock()һ��
		//AutoDiffCostFunctionΪģ���࣬��ceres���Ƽ��Ŀ��Զ��󵼵Ĵ��ۺ������乹�캯��Ϊ
		//ceres::AutoDiffCostFunction<CostFunctor, int residualDim, int paramDim>(CostFunctor* functor);
		//����CostFunctorΪ�º���������Ϊstruct����class�����ڴ�����֪�����Ͷ���в�ļ�����̡�����ķº���ΪReprojectionError3D�ṹ��
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