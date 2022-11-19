#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>

#include "../factor/integration_base.h"
#include "../utility/utility.h"
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
public:
    ImageFrame(){};
    //��7ά��Ӧ������ά���꣬�������꣬�����ٶȣ���x��y���򣩣�
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    // ����map�е�һ��intָ���ǵ��š��ڶ���int�������Ż���֡��ţ����һ����ÿ��vector����ֻ����һ��pairԪ�أ�һ���۲�ֵ��
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;
    Matrix3d R;
    Vector3d T;
    IntegrationBase *pre_integration;
    bool is_key_frame;
};
//xΪ�߶�����
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);