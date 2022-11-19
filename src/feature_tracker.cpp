#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        //255是矩阵mask的每个位置上的值
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time 临时变量，优先保留那些被追踪次数更多的点
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        //track_cnt表示特征点被追踪成功的次数
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    //sort函数第三个参数为Lambda表达式（隐函数），作为可调用（callable)对象被含有范围操作的stl类成员函数所调用，
    //表示将容器内的元素（按照sort的规则）作为参数传给隐函数
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        //如果该矩阵内某点处的值为255，则将这些点保留下来
        //cnt_pts_id这个对象只是起了保存和排序的作用，但是如果只是根据mask的值来判断，那何必要排序？排序的目的是保证forw_pts中的点也是按照被跟踪次数从大到小排列的
        if (mask.at<uchar>(it.second.first) == 255)
        {
            //forw_pts表示当前帧中追踪成功且被保留下来（通过mask）的点集
            forw_pts.push_back(it.second.first);
            //当前帧中追踪成功的点的id
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //circle函数用于在指定图像中的指定位置处画指定位置的圆，颜色和是否填充均可设定
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

//这个函数是用于向追踪队列中添加在当前图像中新检测出来的角点
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

//此应为追踪线程的起始函数
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    //Contrast Limited Adaptive Histogram Equalization 对比度受限自适应直方图均衡，用于增强图像中的整体对比度（直方图分布越均匀，对比度越高？还是细节越好？）
    if (EQUALIZE)
    {
        //3.0为自适应处理中的阈值，Size为分割处理的子图的大小（分成子图是为了加速计算）
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
    //下面这个条件语句是什么意思呢？？表示初始帧操作策略吗？那应该是用cur_img来进行判断吧？
    if (forw_img.empty())
    {
        //prev_img这个量有啥用?说上一帧图像（已检测出特征点或以光流追踪出点），用于最新帧的光流
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }
    //forw表示当前最新帧（即将要处理的），cur表示的是已被处理的上一帧，prev应该是更前一帧
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //此为稀疏光流跟踪。光流追踪的前提是：1. 对象的像素强度在连续帧之间不会改变；2. 相邻像素具有相似的运动。
        // forw_pts就是要求解的下一图像中的特征点；status为状态向量，被跟踪成功的点的状态为1；err为错误向量；Size为搜索窗口大小，3为金字塔最大层数（4层，从0开始计算）
        //ststus的元素数量应该和cur_pts的元素数量一样
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            //若被成功跟踪的点位于边界之外，则删除
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //根据后面的赋值操作，此处prev_pts的初始长度和cur_pts（以及forw_pts）的初始长度不一样了。cur_pts和forw_pts的初始长度均为MAX_CNT
        //prev_pts和status的长度不一样，但是没关系，status的前面元素和prev_pts中的元素是id对应的，这会去除prev_pts中没被forw_img中跟踪到的点
        //cur_pts中会被去除的点更多，因为它还包含中上一帧中重新检测的点，这些点在prev_pts中是没有的
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    //被成功追踪的点，被追踪次数+1
    for (auto &n : track_cnt)
        n++;
    //只有需要publish的帧，才会检测新的特征点以达到规定数量！！！
    if (PUB_THIS_FRAME)
    {
        //用重投影误差估计的方法来计算基础矩阵，并且只保留内点
        rejectWithF();
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        //ROS_DEBUG("set mask costs %fms", t_m.toc());

        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //这里mask应该是用来给定已经被追踪成功的点，防止它们被重新检测
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    //prev_pts初始长度和cur_pts也是不一样的，cur_pts中多了上一帧中重新检测的点中被最新帧成功追踪的那部分点
    //为什么总是只保存三帧的数据呢？
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    //经过上面的reduce（两者同大小）和addPoints之后（forw_pts增加），cur_pts（包括上面已赋值的prev_pts）和forw_pts的长度已不同，两者可以直接赋值吗？？可以
    //将一个vector赋值给另外一个vector时，是拷贝。注意，左侧的vector无论之前什么长度，都会变成跟右侧vector长度相等，元素相等
    cur_pts = forw_pts;
    //cur_un_pts的更新在下面这个函数里，需要用到更新之后的cur_pts（带畸变的像素坐标）
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        //此时cur_pts和forw_pts的点数已经是一样的了，因为前面已经删除过不在最新图像追踪成功的点
        //un_cur_pts和un_forw_pts
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            //tmp_p为临时创建的变量，存储计算得到的去畸变归一化坐标，目的是为了找到基础矩阵和排除外点
            //这个去畸变的过程不影响特征点队列cur_pts等中的元素
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            //再将去畸变归一化坐标转化到像素坐标，以便作为findFundamentalMat函数的输入
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //un_cur_pts和un_forw_pts去畸变的图像点
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        //ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    //由于n_id是静态变量，因此id的值最大不止MAX_CNT,而是从一开始到最新图像中所有被检测到的点的数量
    if (i < ids.size())
    {
        //id为-1的点是当前帧新检测出来的点，在已有的点数（n_id-1）的基础上进行编号
        //即使在最新的特征点队列中，某些旧的特征点已跟丢，但是它的id仍然在，新的点id只能累加
        if (ids[i] == -1)
            //静态变量，可以记录。从0开始编号
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    //instance()是静态函数，静态函数可以使用类名和域限定符::直接调用，返回的是智能指针，因此用->访问所指向的对象中的成员函数
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    //undistortedp的长度为ROW*COL
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

//注意，对相机进行去畸变是在 归一化平面上 进行的，最终得到的是去畸变的归一化坐标
void FeatureTracker::undistortedPoints()
{
    //cur_un_pts为归一化相机坐标系下的去畸变坐标
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        //从liftProjective的函数定义可知，a是带畸变的图像像素坐标（原始图像），b就是求解的无畸变的归一化坐标（函数里严格保证了z=1），为什么这里还要除以z？
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        //pts_velocity为当前帧相对前一帧特征点沿 x,y方向的像素移动速度
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            //prev_un_pts_map和cur_un_pts_map的长度是一样的（从此函数最后的语句可看出，都是MAX_CNT）
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                //这个条件表示在prev_un_pts_map找到相同id的点
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    //这种情况存在吗？既不是最新帧中重新检测的点，又不是从上一帧中追踪过来的？
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
