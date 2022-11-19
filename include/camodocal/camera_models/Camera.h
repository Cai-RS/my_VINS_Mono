#ifndef CAMERA_H
#define CAMERA_H

#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <vector>

namespace camodocal
{

class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum ModelType
    {
        KANNALA_BRANDT,
        MEI,
        PINHOLE,
        SCARAMUZZA
    };

    //类中类作为Camera类的底层实现。此Parameters为抽象类，不能创建对象
    class Parameters
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Parameters(ModelType modelType);

        Parameters(ModelType modelType, const std::string& cameraName,
                   int w, int h);

        //下面这四个也是函数，返回类型int&表示指向整形空间的引用，因此可通过此函数修改对象内的成员
        ModelType& modelType(void);
        std::string& cameraName(void);
        int& imageWidth(void);
        int& imageHeight(void);

        //下面送个为常成员函数，不能修改对象内的任何成员，只能读不能写，也不能调用非常成员函数，但可以调用其他的常成员函数
        ModelType modelType(void) const;
        const std::string& cameraName(void) const;
        int imageWidth(void) const;
        int imageHeight(void) const;

        int nIntrinsics(void) const;

        //在基类中不能对虚函数给出有意义的实现，而把它声明为纯虚函数，它的实现留给该基类的派生类去做
        //包含纯虚函数的类称为抽象类,由于抽象类包含了没有定义的纯虚函数，所以不能定义抽象类的对象,即不能定义Parameters类的对象
        virtual bool readFromYamlFile(const std::string& filename) = 0;
        virtual void writeToYamlFile(const std::string& filename) const = 0;

    protected:
        ModelType m_modelType;
        int m_nIntrinsics;
        std::string m_cameraName;
        int m_imageWidth;
        int m_imageHeight;
    };

    // 为什么外类中也要有这些函数，且为纯虚函数？
    virtual ModelType modelType(void) const = 0;
    virtual const std::string& cameraName(void) const = 0;
    virtual int imageWidth(void) const = 0;
    virtual int imageHeight(void) const = 0;

    virtual cv::Mat& mask(void);
    virtual const cv::Mat& mask(void) const;

    //参数均为常量引用
    virtual void estimateIntrinsics(const cv::Size& boardSize,
                                    const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                    const std::vector< std::vector<cv::Point2f> >& imagePoints) = 0;
    //这里的外参指的是什么？又不是双目相机
    virtual void estimateExtrinsics(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    cv::Mat& rvec, cv::Mat& tvec) const;

    // Lift points from the image plane to the sphere 将图像点投影到球面
    virtual void liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const = 0;
    //%output P

    // Lift points from the image plane to the projective space 将图像点逆投影到三维空间
    virtual void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const = 0;
    //%output P

    // Projects 3D points to the image plane (Pi function) 正向投影
    virtual void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const = 0;
    //%output p

    // Projects 3D points to the image plane (Pi function)
    // and calculates jacobian
    //virtual void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
    //                          Eigen::Matrix<double,2,3>& J) const = 0;
    //%output p
    //%output J

    //图像去畸变
    virtual void undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const = 0;
    //%output p

    //virtual void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale = 1.0) const = 0;
    virtual cv::Mat initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                            float fx = -1.0f, float fy = -1.0f,
                                            cv::Size imageSize = cv::Size(0, 0),
                                            float cx = -1.0f, float cy = -1.0f,
                                            cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const = 0;

    virtual int parameterCount(void) const = 0;

    virtual void readParameters(const std::vector<double>& parameters) = 0;
    virtual void writeParameters(std::vector<double>& parameters) const = 0;

    virtual void writeParametersToYamlFile(const std::string& filename) const = 0;

    virtual std::string parametersToString(void) const = 0;

    /**
     * \brief Calculates the reprojection distance between points
     *
     * \param P1 first 3D point coordinates
     * \param P2 second 3D point coordinates
     * \return euclidean distance in the plane 欧氏距离
     */
    double reprojectionDist(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2) const;

    double reprojectionError(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                             const std::vector< std::vector<cv::Point2f> >& imagePoints,
                             const std::vector<cv::Mat>& rvecs,
                             const std::vector<cv::Mat>& tvecs,
                             cv::OutputArray perViewErrors = cv::noArray()) const;

    double reprojectionError(const Eigen::Vector3d& P,
                             const Eigen::Quaterniond& camera_q,
                             const Eigen::Vector3d& camera_t,
                             const Eigen::Vector2d& observed_p) const;

    void projectPoints(const std::vector<cv::Point3f>& objectPoints,
                       const cv::Mat& rvec,
                       const cv::Mat& tvec,
                       std::vector<cv::Point2f>& imagePoints) const;
protected:
    cv::Mat m_mask;
};

typedef boost::shared_ptr<Camera> CameraPtr;
//常量对象的智能指针（C++不允许在常量对象上调用成员函数，除非成员函数本身也被声明为常量，因此上面重载的常量函数这里就可以被调用）
typedef boost::shared_ptr<const Camera> CameraConstPtr;

}

#endif
