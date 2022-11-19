#include "System.h"
//Pangolin是一个基于OpenGL的轻量级开源绘图库，在许多开源SLAM算法（例如ORB-SLAM）中都会用来进行可视化操作
// https://blog.csdn.net/weixin_43991178/article/details/105119610
#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace pangolin;

System::System(string sConfig_file_)
    :bStart_backend(true)
{
    string sConfig_file = sConfig_file_ + "euroc_config.yaml";

    cout << "1 System() sConfig_file: " << sConfig_file << endl;
    //parameters.cpp中的函数，用于从配置文件读取所有的系统参数（均为全局变量）
    readParameters(sConfig_file);
    //0表示第一个相机，trackerData中元素为FeatureTracker类，其中包含有一个成员为Camera类的智能指针
    //成员函数readIntrinsicParameter用于读取配置文件并返回一个子类Camera的指针给Camera类的智能指针（虚继承）
    trackerData[0].readIntrinsicParameter(sConfig_file);

    estimator.setParameter();
    ofs_pose.open("./pose_output.txt",fstream::app | fstream::out);
    if(!ofs_pose.is_open())
    {
        cerr << "ofs_pose is not open" << endl;
    }
    // thread thd_RunBackend(&System::process,this);
    // thd_RunBackend.detach();
    cout << "2 System() end" << endl;
}

System::~System()
{
    bStart_backend = false;
    
    pangolin::QuitAll();
    
    //buf线程
    m_buf.lock();
    while (!feature_buf.empty())
        //pop时会queue中的元素（struct的智能指针）给消除，会调用struct的析构函数释放相关内存？
        feature_buf.pop();
    while (!imu_buf.empty())
        imu_buf.pop();
    m_buf.unlock();

    //estimator线程
    m_estimator.lock();
    estimator.clearState();
    m_estimator.unlock();

    ofs_pose.close();
}

void System::PubImageData(double dStampSec, Mat &img)
{
    if (!init_feature)
    {
        cout << "1 PubImageData skip the first detected feature, which doesn't contain optical flow speed" << endl;
        init_feature = 1;
        return;
    }

    if (first_image_flag)
    {
        cout << "2 PubImageData first_image_flag" << endl;
        first_image_flag = false;
        first_image_time = dStampSec;
        last_image_time = dStampSec;
        return;
    }
    // detect unstable camera stream 两帧相隔时间太长 或者 出错了
    if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time)
    {
        cerr << "3 PubImageData image discontinue! reset the feature tracker!" << endl;
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        return;
    }
    last_image_time = dStampSec;
    // frequency control  发布频率不能太频繁
    if (round(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (dStampSec - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            //这个量用于计算一段时间内发布图像的频率
            first_image_time = dStampSec;
            pub_count = 0;
        }
    }
    else
        //发布频率太高，parameters.h中的参数
    {
        PUB_THIS_FRAME = false;
    }

    TicToc t_r;
    // cout << "3 PubImageData t : " << dStampSec << endl;
    //此函数用于读入图像，预处理和光流追踪，但是最新图像中的追踪点队列中的id还没进行编号
    trackerData[0].readImage(img, dStampSec);

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        completed |= trackerData[0].updateID(i);

        if (!completed)
            break;
    }
    //只有需要publish的帧，前面readImage()函数里才会检测新的特征点以达到规定数量！！！
    if (PUB_THIS_FRAME)
    {
        pub_count++;
        shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        //使用关联式容器set存储的各个键值对，要求键key和值value必须相等。使用set容器存储的各个元素的值必须各不相同
        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //cur_un_pts表示当前最新帧中已处理过后（光流追踪+RANSAC去外点+新检测）的去畸变的特征点的归一化坐标(只保留前两维）
            auto &un_pts = trackerData[i].cur_un_pts;
            //cur_pts最新帧中已处理的像素坐标
            auto &cur_pts = trackerData[i].cur_pts;
            //最新帧中已处理的特征点击的id
            auto &ids = trackerData[i].ids;
            //特征点的像素速度
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->points.push_back(Vector3d(x, y, z));
                    //点的序号如何编排？
                    //trackerData每个相机的特征点ids都是从0开始
                    //而在id_of_point中是每个相机靠前的ids排在前面，即先排进每个相机ids=0的点，再排进ids=1的点...
                    //但这样id_of_point和points是否不对应了？？只是点在vector中的索引与其id值不对应，但两个vector中相同位置处都是同一点的信息
                    //实际上不可能是多个相机，如果是多个相机，何不直接变成多目VIO？因此这里实际上默认只有一个相机
                    feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);
                    feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }
            //}
            // skip the first image; since no optical speed on frist image
            if (!init_pub)
            {
                cout << "4 PubImage init_pub skip the first image!" << endl;
                init_pub = 1;
            }
            else
            {
                m_buf.lock();
                feature_buf.push(feature_points);
                // cout << "5 PubImage t : " << fixed << feature_points->header
                //     << " feature_buf size: " << feature_buf.size() << endl;
                m_buf.unlock();
                con.notify_one();
            }
        }
    }

#ifdef __linux__
    cv::Mat show_img;
	cv::cvtColor(img, show_img, CV_GRAY2RGB);
	if (SHOW_TRACK)
	{
		for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++)
        {
            //根据被追踪的次数来决定特征点标记颜色的深浅
			double len = min(1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE);
			cv::circle(show_img, trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
		}

        cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
		cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
	}
#endif    
    // cout << "5 PubImage" << endl;
    
}

//两帧图像之间有许多的imu数据。此函数是从队列imu_buf和feature_buf中取出所有有效的测量信息并删掉（pop），加入变量measurements中
vector<pair<vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements()
{
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

    while (true)
    {
        //队列中"有效"的信息对被取光了，返回measurements
        if (imu_buf.empty() || feature_buf.empty())
        {
            // cerr << "1 imu_buf.empty() || feature_buf.empty()" << endl;
            return measurements;
        }
        //td是经过优化估计的imu和相机之间的帧时间差（前者-后者）
        //(此项目中没有要对其进行估计，因此为初始设定值0.0）
        // 在队列中，先进先出，元素被插入的一端被称为“ back”，并从被称为“ front”的另一端删除。函数front()返回最旧元素的引用
        //如果最新的imu时刻都不大于最早的图像时刻，则需要等待更加新的imu信息
        if (!(imu_buf.back()->header > feature_buf.front()->header + estimator.td))
        {
            cerr << "wait for imu, only should happen at the beginning sum_of_wait: " 
                << sum_of_wait << endl;
            sum_of_wait++;
            return measurements;
        }
        //如果最早的imu时刻不小于最早的图像时刻，则抛弃掉这帧最老的图像（因为得不到完整的预积分）。时间上“等于”的帧也需要删除。初始时刻的速度应该是多少呢？通过首两帧之间的速度来估计？
        if (!(imu_buf.front()->header < feature_buf.front()->header + estimator.td))
        {
            cerr << "throw img, only should happen at the beginning" << endl;
            feature_buf.pop();
            continue;
        }
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        vector<ImuConstPtr> IMUs;
        //得到当前帧和上一帧之间的imu数据。为什么不能是等于？
        while (imu_buf.front()->header < img_msg->header + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // cout << "1 getMeasurements IMUs size: " << IMUs.size() << endl;
        //大于或等于的那一时刻imu在这里添加了
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty()){
            cerr << "no imu between two image" << endl;
        }
        // cout << "1 getMeasurements img t: " << fixed << img_msg->header
        //     << " imu begin: "<< IMUs.front()->header 
        //     << " end: " << IMUs.back()->header
        //     << endl;
        //每个元素是一帧图像和相关的imu数据（imu数据不早于该图像，且不晚于上一帧图像）。最终穷尽imu_buf或feature_buf所有满足条件的信息对
        measurements.emplace_back(IMUs, img_msg);
    }
    //如果最老帧是初始帧图像，那这些比初始帧还老的imu数据会怎么处理？？见ProcessBackEnd()函数
    return measurements;
}

//将得到的imu数据插入imu_buf队列中
void System::PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
    const Eigen::Vector3d &vAcc)
{
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
	imu_msg->header = dStampSec;
	imu_msg->linear_acceleration = vAcc;
	imu_msg->angular_velocity = vGyr;

    if (dStampSec <= last_imu_t)
    {
        cerr << "imu message in disorder!" << endl;
        return;
    }
    last_imu_t = dStampSec;
    // cout << "1 PubImuData t: " << fixed << imu_msg->header
    //     << " acc: " << imu_msg->linear_acceleration.transpose()
    //     << " gyr: " << imu_msg->angular_velocity.transpose() << endl;
    m_buf.lock();
    imu_buf.push(imu_msg);
    // cout << "1 PubImuData t: " << fixed << imu_msg->header 
    //     << " imu_buf size:" << imu_buf.size() << endl;
    m_buf.unlock();
    con.notify_one();
}

// thread: visual-inertial odometry
void System::ProcessBackEnd()
{
    //这句话应该放在循环里面比较合适？
    cout << "1 ProcessBackEnd start" << endl;
    while (bStart_backend)
    {
        // cout << "1 process()" << endl;
        vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;
        
        unique_lock<mutex> lk(m_buf);
        //带条件的被阻塞：wait函数设置了谓词(Predicate)，这里的谓词使用的是Lambda函数（&表示为隐式引用捕获，即自动捕获变量measurements的引用）
        //只有当pred条件为false时调用该wait函数才会阻塞当前线程，并且在收到其它线程的通知后只有当pred为true时才会被解除阻塞。
        //只有当一开始measurements的维度为0时，wait才会执行，unlock，当前线程阻塞。直到measurements的维度重新不为0，才会lock，继续线程。
        //wait最多只会被执行一次，如果unlock的条件不满足，则会继续往下执行。
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        //wakeup了不一定代表wait中的条件已经被满足（被称为“假醒”，概率比较小），因此需要额外判断条件是否真的满足了
        //为什么要大于1？
        if( measurements.size() > 1){
        cout << "1 getMeasurements size: " << measurements.size() 
            //两帧图像之间的imu次数
            << " imu sizes: " << measurements[0].first.size()
            //测量信息队列中还剩余的信息个数
            << " feature_buf size: " <<  feature_buf.size()
            << " imu_buf size: " << imu_buf.size() << endl;
        }
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator.td;
                if (t <= img_t)
                {
                    //表示此时为初始帧，若为初始帧，则dt=0
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x();
                    dy = imu_msg->linear_acceleration.y();
                    dz = imu_msg->linear_acceleration.z();
                    rx = imu_msg->angular_velocity.x();
                    ry = imu_msg->angular_velocity.y();
                    rz = imu_msg->angular_velocity.z();
                    //若dt=0，则此时给进去的数据会用于创建首个预积分对象（首帧图像之前时刻的imu数据），但这个预积分量不会被更新计算
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("1 BackEnd imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else
                {
                    //如果某个imu的时刻比纠正同步（此同步是指纠正数据传输速度差异导致的时差）的相机时刻要晚。这是有可能的，因为在形成measurements中的元素时，允许每个imu集最后一个时刻大于等于相应相机的时刻
                    //此时的current_time应该是倒数第二个imu的时间点，dt_1为倒数第二个imu到相机的时间差
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    //最后两个imu之间的时间差要大于0
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    //线性插值，得到相机的时刻的imu数据
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // cout << "processing vision data with stamp:" << img_msg->header 
            //     << " img_msg->points.size: "<< img_msg->points.size() << endl;

            // TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++) 
            {
                //为什么要+0.5？？非整数可以取余运算吗？？
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                //归一化坐标
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                //如果相机个数大于1，那这里不是把所有相机观测的点都放进来这一幅图像吗？每个相机检测到的特征点集都是相同的吗？
                //但是map中key值是不可重复的，因此这里相同feature_id的value会被覆盖？但是如何能保证不同相机中相同feature_id的点是同个路标点？？
                //默认只有一个相机吗？
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            TicToc t_processImage;
            estimator.processImage(image, img_msg->header);
            //如果VIO初始化已完成，则意味着已完成一次后端BA优化
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                Vector3d p_wi;
                Quaterniond q_wi;
                //已完成优化估计的最新帧时刻imu的位姿
                q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);
                p_wi = estimator.Ps[WINDOW_SIZE];
                vPath_to_draw.push_back(p_wi);
                double dStamp = estimator.Headers[WINDOW_SIZE];
                //如果一个数字太大，无法使用 setprecision 指定的有效数位数来打印，则许多系统会自动以科学表示法的方式打印，
                //为了防止出现这种情况，可以使用另一个流操作符 fixed，它表示浮点输出应该以固定点或小数点表示法显示。
                // https://blog.csdn.net/u011754972/article/details/121752238
                cout << "1 BackEnd processImage dt: " << fixed << t_processImage.toc() << " stamp: " <<  dStamp << " p_wi: " << p_wi.transpose() << endl;
                //写入位姿结果的文件
                ofs_pose << fixed << dStamp << " " << p_wi.transpose() << " " << q_wi.coeffs().transpose() << endl;
            }
        }
        m_estimator.unlock();
    }
}

void System::Draw() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // 0.1是相机的最近视距，1000是最远视距
    // https://blog.csdn.net/weixin_43991178/article/details/105119610?spm=1001.2014.3001.5502
    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    // pangolin::OpenGlRenderState s_cam(
    //         pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
    //         pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    // );

    // pangolin::View &d_cam = pangolin::CreateDisplay()
    //         .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
    //         .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
         
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points 这里的点指的是相机的位置点
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

#ifdef __APPLE__
void System::InitDrawGL() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
}

void System::DrawGLFrame() 
{  

    if (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
            
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
#endif

}
