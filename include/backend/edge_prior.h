//
// Created by heyijia on 19-1-30.
//
//这个文件并没有被使用

#ifndef SLAM_COURSE_EDGE_PRIOR_H
#define SLAM_COURSE_EDGE_PRIOR_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"


namespace myslam {
namespace backend {

/**
* EdgeSE3Prior，此边为 1 元边，与之相连的顶点有：Ti。为什么只有位姿有先验？？先验不应该是上个时刻的marg剩余吗？
*/
class EdgeSE3Prior : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3Prior(const Vec3 &p, const Qd &q) :
            Edge(6, 1, std::vector<std::string>{"VertexPose"}),
            Pp_(p), Qp_(q) {}

    /// 返回边的类型信息
    virtual std::string TypeInfo() const override { return "EdgeSE3Prior"; }

    /// 计算残差
    virtual void ComputeResidual() override;

    /// 计算雅可比
    virtual void ComputeJacobians() override;


private:
    //下面这俩是先验值，点中存储的是待估计值，两者相减为残差
    Vec3 Pp_;   // pose prior
    Qd   Qp_;   // Rotation prior
};

}
}


#endif //SLAM_COURSE_EDGE_PRIOR_H
