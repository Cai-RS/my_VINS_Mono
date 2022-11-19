#include "initial/initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    // �������imu֮�����ת�����ʼ��Ϊ��λ����
    ric = Matrix3d::Identity();
}
 // corres�洢���ǵ�k��k+1֡ͼ���е�ƥ��㣬���Լ�����ת����
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    // ric��ʾ�����imu֮�����ת�������ʽ�Ӻ͵�28���е�r1�����Ԫ���Ĳ��죩������в�ֱ���imu������������ת��ric�������R_bk_c+1������òв
    //CalibrationExRotation�����α궨����ֻ����ESTIMATE_EXTRINSIC==2ʱ�Ż���ã�ric���Ƶĳ�ʼֵ�ǵ�λ����tic��ʼֵ��0����parameters.cpp
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);

    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        //ROS_DEBUG("%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        // ����Ϊ��Ԫ����˺��ҳ�ʱ�ľ���
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }
    // ��֡������1ʱ��A������ǳ����ģ�������������4�����������Ƿ񳬶���Aһ�㶼�����ȵģ���û����ռ䣩�������Ҫ����С���ˣ�������С������svd�����
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    //tail(n)��ʾȡ���n��Ԫ��
    ric_cov = svd.singularValues().tail<3>();

    //���ڵ�һ���ж������Ĵ��ڣ����������εĺ����Ƕ�ÿһ֡��Ҫ����ģ���ֻ���ܻ᷵�����һ�μ���Ľ���������Ǽ��ϻ����� ���е�ǰ��֡ƥ���� ������εļ��㣬��Ϊ������ȷ�����
    // �ڶ����ж������Ӻζ�����Ϊʲô�����ڶ�С������ֵ����0.25�Ϳ��ԣ�
    //ֻ�е��ڶ�����������ʱ�����յõ��Ľ���ſ����㹻��ȷ������Բ���������������Ϊ��ι���ʧ��
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    //�˵㷨����������
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);
        //E��-E���ڶԼ�Լ����������������Ҫ������ֽ�õ���R�����ʣ���R������֮һ��������ʽ����1
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        // ����Ϊɶ��ֱ��ת��ans_R_cv��ֵ��ans_R_eigen�أ�
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        //����Ӧ�þ���ȡ�����������ϵ�µ������zֵ
        double normal_factor = pointcloud.col(i).at<float>(3);
        //Ϊʲô�����ڱ任���������ϵ��ʱҪ�ȳ���zֵ�أ�
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    //ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    // ��Ҫ�����еĵ�����֡�е���ȶ�Ϊ0������Ϊʲô��
    return 1.0 * front_count / pointcloud.cols;
}

void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    //��ʾ��z����ת90��ľ���
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    //��ʾ��z����ת-90��ľ���
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    // �����λ�ƵĹ�ʽ��ʮ�Ľ��еĲ�ͬ��Ӧ���Ǽ��ˣ�
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
