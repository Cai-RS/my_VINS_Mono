#include "initial/initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    // 将相机和imu之间的旋转矩阵初始化为单位矩阵
    ric = Matrix3d::Identity();
}
 // corres存储的是第k和k+1帧图像中的匹配点，用以计算旋转矩阵
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    // ric表示相机和imu之间的旋转矩阵，这个式子和第28行中的r1作差（四元数的差异）即求出残差（分别用imu和相机的相对旋转与ric算出两个R_bk_c+1，作差得残差）
    //CalibrationExRotation这个外参标定函数只会在ESTIMATE_EXTRINSIC==2时才会调用，ric估计的初始值是单位矩阵，tic初始值是0，见parameters.cpp
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
        // 这俩为四元数左乘和右乘时的矩阵
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
    // 当帧数大于1时，A矩阵就是超定的（行数大于列数4）。但不论是否超定，A一般都是满秩的（即没有零空间），因此需要用最小二乘，这里最小二乘用svd来求解
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    //tail(n)表示取最后n个元素
    ric_cov = svd.singularValues().tail<3>();

    //由于第一个判断条件的存在，这个计算外参的函数是对每一帧都要运算的，但只可能会返回最后一次计算的结果。最终是集合滑窗中 所有的前后帧匹配点对 进行外参的计算，因为这样精确度最高
    // 第二个判断条件从何而来？为什么倒数第二小的奇异值大于0.25就可以？
    //只有当第二个条件满足时，最终得到的结果才可能足够精确。如果仍不满足条件，则认为外参估计失败
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
    //八点法求解基础矩阵
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
        //E和-E对于对极约束均成立，但是需要先满足分解得到的R的性质，而R的性质之一就是行列式等于1
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        // 这里为啥不直接转置ans_R_cv后赋值给ans_R_eigen呢？
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
        //这里应该就是取点的世界坐标系下的坐标的z值
        double normal_factor = pointcloud.col(i).at<float>(3);
        //为什么这里在变换到相机坐标系下时要先除以z值呢？
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    //ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    // 不要求所有的点在两帧中的深度都为0，这是为什么？
    return 1.0 * front_count / pointcloud.cols;
}

void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    //表示沿z轴旋转90°的矩阵
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    //表示沿z轴旋转-90°的矩阵
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    // 这个求位移的公式和十四讲中的不同，应该是简化了？
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
