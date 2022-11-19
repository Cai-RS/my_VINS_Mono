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
        //255�Ǿ���mask��ÿ��λ���ϵ�ֵ
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time ��ʱ���������ȱ�����Щ��׷�ٴ�������ĵ�
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        //track_cnt��ʾ�����㱻׷�ٳɹ��Ĵ���
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    //sort��������������ΪLambda���ʽ��������������Ϊ�ɵ��ã�callable)���󱻺��з�Χ������stl���Ա���������ã�
    //��ʾ�������ڵ�Ԫ�أ�����sort�Ĺ�����Ϊ��������������
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        //����þ�����ĳ�㴦��ֵΪ255������Щ�㱣������
        //cnt_pts_id�������ֻ�����˱������������ã��������ֻ�Ǹ���mask��ֵ���жϣ��Ǻα�Ҫ���������Ŀ���Ǳ�֤forw_pts�еĵ�Ҳ�ǰ��ձ����ٴ����Ӵ�С���е�
        if (mask.at<uchar>(it.second.first) == 255)
        {
            //forw_pts��ʾ��ǰ֡��׷�ٳɹ��ұ�����������ͨ��mask���ĵ㼯
            forw_pts.push_back(it.second.first);
            //��ǰ֡��׷�ٳɹ��ĵ��id
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //circle����������ָ��ͼ���е�ָ��λ�ô���ָ��λ�õ�Բ����ɫ���Ƿ��������趨
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

//���������������׷�ٶ���������ڵ�ǰͼ�����¼������Ľǵ�
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

//��ӦΪ׷���̵߳���ʼ����
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    //Contrast Limited Adaptive Histogram Equalization �Աȶ���������Ӧֱ��ͼ���⣬������ǿͼ���е�����Աȶȣ�ֱ��ͼ�ֲ�Խ���ȣ��Աȶ�Խ�ߣ�����ϸ��Խ�ã���
    if (EQUALIZE)
    {
        //3.0Ϊ����Ӧ�����е���ֵ��SizeΪ�ָ�����ͼ�Ĵ�С���ֳ���ͼ��Ϊ�˼��ټ��㣩
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
    //����������������ʲô��˼�أ�����ʾ��ʼ֡������������Ӧ������cur_img�������жϰɣ�
    if (forw_img.empty())
    {
        //prev_img�������ɶ��?˵��һ֡ͼ���Ѽ�����������Թ���׷�ٳ��㣩����������֡�Ĺ���
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }
    //forw��ʾ��ǰ����֡������Ҫ����ģ���cur��ʾ�����ѱ��������һ֡��prevӦ���Ǹ�ǰһ֡
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //��Ϊϡ��������١�����׷�ٵ�ǰ���ǣ�1. ���������ǿ��������֮֡�䲻��ı䣻2. �������ؾ������Ƶ��˶���
        // forw_pts����Ҫ������һͼ���е������㣻statusΪ״̬�����������ٳɹ��ĵ��״̬Ϊ1��errΪ����������SizeΪ�������ڴ�С��3Ϊ��������������4�㣬��0��ʼ���㣩
        //ststus��Ԫ������Ӧ�ú�cur_pts��Ԫ������һ��
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            //�����ɹ����ٵĵ�λ�ڱ߽�֮�⣬��ɾ��
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //���ݺ���ĸ�ֵ�������˴�prev_pts�ĳ�ʼ���Ⱥ�cur_pts���Լ�forw_pts���ĳ�ʼ���Ȳ�һ���ˡ�cur_pts��forw_pts�ĳ�ʼ���Ⱦ�ΪMAX_CNT
        //prev_pts��status�ĳ��Ȳ�һ��������û��ϵ��status��ǰ��Ԫ�غ�prev_pts�е�Ԫ����id��Ӧ�ģ����ȥ��prev_pts��û��forw_img�и��ٵ��ĵ�
        //cur_pts�лᱻȥ���ĵ���࣬��Ϊ������������һ֡�����¼��ĵ㣬��Щ����prev_pts����û�е�
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    //���ɹ�׷�ٵĵ㣬��׷�ٴ���+1
    for (auto &n : track_cnt)
        n++;
    //ֻ����Ҫpublish��֡���Ż����µ��������Դﵽ�涨����������
    if (PUB_THIS_FRAME)
    {
        //����ͶӰ�����Ƶķ���������������󣬲���ֻ�����ڵ�
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
            //����maskӦ�������������Ѿ���׷�ٳɹ��ĵ㣬��ֹ���Ǳ����¼��
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
    //prev_pts��ʼ���Ⱥ�cur_ptsҲ�ǲ�һ���ģ�cur_pts�ж�����һ֡�����¼��ĵ��б�����֡�ɹ�׷�ٵ��ǲ��ֵ�
    //Ϊʲô����ֻ������֡�������أ�
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    //���������reduce������ͬ��С����addPoints֮��forw_pts���ӣ���cur_pts�����������Ѹ�ֵ��prev_pts����forw_pts�ĳ����Ѳ�ͬ�����߿���ֱ�Ӹ�ֵ�𣿣�����
    //��һ��vector��ֵ������һ��vectorʱ���ǿ�����ע�⣬����vector����֮ǰʲô���ȣ������ɸ��Ҳ�vector������ȣ�Ԫ�����
    cur_pts = forw_pts;
    //cur_un_pts�ĸ�������������������Ҫ�õ�����֮���cur_pts����������������꣩
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        //��ʱcur_pts��forw_pts�ĵ����Ѿ���һ�����ˣ���Ϊǰ���Ѿ�ɾ������������ͼ��׷�ٳɹ��ĵ�
        //un_cur_pts��un_forw_pts
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            //tmp_pΪ��ʱ�����ı������洢����õ���ȥ�����һ�����꣬Ŀ����Ϊ���ҵ�����������ų����
            //���ȥ����Ĺ��̲�Ӱ�����������cur_pts���е�Ԫ��
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            //�ٽ�ȥ�����һ������ת�����������꣬�Ա���ΪfindFundamentalMat����������
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //un_cur_pts��un_forw_ptsȥ�����ͼ���
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
    //����n_id�Ǿ�̬���������id��ֵ���ֹMAX_CNT,���Ǵ�һ��ʼ������ͼ�������б���⵽�ĵ������
    if (i < ids.size())
    {
        //idΪ-1�ĵ��ǵ�ǰ֡�¼������ĵ㣬�����еĵ�����n_id-1���Ļ����Ͻ��б��
        //��ʹ�����µ�����������У�ĳЩ�ɵ��������Ѹ�������������id��Ȼ�ڣ��µĵ�idֻ���ۼ�
        if (ids[i] == -1)
            //��̬���������Լ�¼����0��ʼ���
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    //instance()�Ǿ�̬��������̬��������ʹ�����������޶���::ֱ�ӵ��ã����ص�������ָ�룬�����->������ָ��Ķ����еĳ�Ա����
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
    //undistortedp�ĳ���ΪROW*COL
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

//ע�⣬���������ȥ�������� ��һ��ƽ���� ���еģ����յõ�����ȥ����Ĺ�һ������
void FeatureTracker::undistortedPoints()
{
    //cur_un_ptsΪ��һ���������ϵ�µ�ȥ��������
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        //��liftProjective�ĺ��������֪��a�Ǵ������ͼ���������꣨ԭʼͼ�񣩣�b���������޻���Ĺ�һ�����꣨�������ϸ�֤��z=1����Ϊʲô���ﻹҪ����z��
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        //pts_velocityΪ��ǰ֡���ǰһ֡�������� x,y����������ƶ��ٶ�
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            //prev_un_pts_map��cur_un_pts_map�ĳ�����һ���ģ��Ӵ˺����������ɿ���������MAX_CNT��
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                //���������ʾ��prev_un_pts_map�ҵ���ͬid�ĵ�
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    //������������𣿼Ȳ�������֡�����¼��ĵ㣬�ֲ��Ǵ���һ֡��׷�ٹ����ģ�
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
