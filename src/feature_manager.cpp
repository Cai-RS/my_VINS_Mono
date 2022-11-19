#include "feature_manager.h"

//这样计算结束帧的序号，前提是帧是连续的，即点的跟踪不能断
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

//对列表对象中的数据进行清空
void FeatureManager::clearState()
{
    feature.clear();
}

//计数被成功追踪的点id数（应该是指滑窗内的）
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();
        //it.start_frame < WINDOW_SIZE - 2 这句的目的是什么？是说点的起始帧必须在整个运行期间的前WINDOW_SIZE - 2帧之内？如果是这样，那么这里所谓运行期间就是每个滑窗内
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

//此函数用于最新帧中的点的信息添加（进feature中），并且检查视差
//frame_count是帧的序号（可以是滑窗中的第一帧吗？这里应该不用担心是第一帧，因为前面几帧属于初始化的进程）；map中第一个int应该是点的id，第二个int是相机id。一个map中可以存储复数元素，像vector一样
//当返回变量是ture时，表明倒数第二帧，即次新帧应该被保留，因为它与最新帧之间的视差足够大
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    //ROS_DEBUG("input feature: %d", (int)image.size());
    //ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts : image)
    {
        //点信息和时间戳，但是为什么始终是second[0]？？这不是永远取vector中的第一个元素吗？可能是因为只有一个元素？
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        //查找STL手册中的find_if函数定义，第三个参数为自定义的predicate function
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
        //上面的find_if函数中，如果找不到符合条件的it，则函数返回的是第二个参数，即feature.end()
        //注意，feature.end()这个迭代器指向位置是list中最后一个元素位置再加1！
        if (it == feature.end())
        {
            //feature是list，里面的点id是按照被观测的时间顺序被放进的，最后的元素就是最新的帧中被观测到的新点
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }
    //要么此帧为头两帧，要么此帧中追踪到的点数小于20（追踪到的点数少，说明最新的两帧之间视差大，不需要后续的视差计算）
    if (frame_count < 2 || last_track_num < 20)
        return true;
    //上面的if是从最新帧和次新帧中特征点的关联数量（last_track_num）来决定次新帧是否关键，如果无法判断，则从次新帧和次次新帧的视差来判断
    for (auto &it_per_id : feature)
    {
        //第一个条件表示起始帧小于或等于倒数第三帧
        //第二个条件表示这个特征点从被观测到开始，到倒数第二帧（等于号）或者倒数第一帧（大于号）一直都被持续地观测到
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            //compensated有偿的，此函数计算倒数第二帧和倒数第三帧之间的视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
    //parallax_num == 0这种情况可能出现吗？？有可能，这样的话倒数第三帧及之前的帧就都会被逐渐地marg掉，因为它们的点从倒数第二帧起已经没法被跟踪了！
    //此时次新帧和次次新帧之间没有关联，当然要保留次新帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        //如果次新帧和最新帧之间视差不够大（last_track_num不够小），且次新帧和次次新帧的视差也不够大，则可以不保留次新帧
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    //ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        assert(it.feature_per_frame.size() != 0);
        assert(it.start_frame >= 0);
        assert(it.used_num >= 0);

        //ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            //ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.used_num == sum);
    }
}

//此函数用于找出在frame_count_l和frame_count_r两帧中都被观察到的点，并且返回所有的点对
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}
 //给定的深度向量x中应该是逆深度，因为estimated_depth是深度
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //只求解了成功追踪且符合要求的点的深度，更新深度时也按照这个条件筛选和赋值
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

//只有那些已经被设置过深度（setDepth函数），并且深度仍然为负的（初始化时的深度就是负的）点才会被移除
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

//跟setDepth函数的区别是什么？
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

//如果参数是写成Vector3d Ps，则Ps只是一个三维向量，但如果后面加了[]，则变为了vector<Vtctor3d>。
//从调用这个函数的命令行来看，传进来的相机到imu的位移参数tic为0
//此特征点三角化的原理见VIO课程“视觉前端”一节的课件，注意这里的待求的点3D坐标是在其观测首帧下的
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //只选择满足追踪成功条件的点来进行三角化
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //如果深度大于0，说明已经被估计过了？
        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        //用来存储每个特征点的三角化的两个公式，每个公式的结果是1*4维，即位姿矩阵的一行
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        //tic[0]是vector<Vector3d>中的第一个元素，但实际上这个vector里可能也就一个元素
        //Ps[i]指的是第i帧时刻的“相机坐标系”到参考帧l“相机坐标系”（或世界坐标系）的位移（表达在参考帧l或世界坐标系下）,
        //但是由于相机和imu之间假设为没有位移（tic=0)，因此Ps也可以说是从第i帧的“imu”坐标系到世界坐标系（第l帧相机坐标系）的位移。如果tic不等于0，则直接让t0=Ps[imu_i]就好，这样t0才能表示使第i帧相机移到世界坐标系
        //Rs指的是该帧时刻的“imu坐标系”到参考帧l“相机坐标系”（或世界坐标系）的旋转，ric则是相机到imu的旋转
        //从下面for循环中的代码来看，Ps应该是某帧相机对应的imu坐标系到初始帧（或世界坐标系）的位移，Rs类似，ric是相机到imu的旋转。
        //Rs是FeatureManager类中的成员，不需要从参数传入
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();
        //计算该点被观察到的其他帧相对该点观测初始帧的位姿，形成参数矩阵P
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            //是从imu_j = imu_i开始的，说明观测初始帧中的观测也能被用来三角化，即使要求解的是该点在观测初始帧下的坐标
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            //r0指向r1，表达在R0中（即从相机帧j变换到该点被观察到的初始帧），这个相对位姿是不受参考帧l选择下计算出来的Ps和Rs的影响的
            //因此后面计算出来的点的深度就是表达在该点被观察到的初始帧坐标系下的深度，VIO的对齐操作对这个值没有影响
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            //这里的归一化不是指Z坐标为1，而是整体三维坐标的模为1，因此下面的P.row(0)和P.row(1)前面要乘以f[2]
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
            //为什么这个不是放在最上面？因为观测初始帧中的观测也能提供有效的两个方程
            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        //待估计量是四维的齐次点坐标，因为最后一位总是1，在优化时把这个约束变为整个向量的模长为1
        //因此这里用svd的方法求解，把最小奇异值对应的右奇异向量作为优化结果，当然还得把第四维归一
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];
        //得到点的深度，这个深度是在初始帧下的
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        //如果估计出来的深度太小，则设为默认深度（值为5.0）。为什么要这么做？
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    // ROS_BREAK();
    return;
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //这个变量i有啥用？？
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

//此函数是把滑窗内最旧帧给marg之后，需要对feature_manager中的特征点信息进行删减和重新排序
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //滑动窗口的第一帧序号为0
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            //选择起始帧中的特征点，因为estimated_depth就是起始帧中点的深度
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            //如果该点仅在最旧帧和次旧帧中被看到，则直接去掉该特征点
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                //将观测首帧的深度信息从最旧帧转移到次旧帧中
                //estimated_depth为深度，那么uv_i应该是归一化坐标
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                //该点在世界坐标系下的坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                //该点在第j帧（次旧帧）相机坐标系下的坐标，new_P为相机坐标系j指向世界坐标系（表达在世界坐标系下）
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        * //WINDOW_SIZE-1是因为从0开始排序
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}
//此函数的调用时机是初始化失败，并且需要marg掉最旧帧时，此时仅进行了sfm估计，没有marg，直接去掉最旧帧信息即可，下一帧进来后继续VIO初始化
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//这里的frame_count指的是新帧进来后的次新帧是第几帧，它的实际id应该为frame_count-1。这里的marg是在最新帧已经加入feature之后再进行的
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        
        //if里面的语句若为真，说明这个特征点的起始帧在要被marg的帧的后一帧，所以这里只需要将它减一即可
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            //被次新帧观测到了，则去掉次新帧中的观测信息
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            //条件若成立，说明这个特征点仅在被marg的次新帧中被观测到，在最新帧中也没有它
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame（对的）
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;
    //为什么倒数第二帧中的point就不需要除以深度？？point本身应该就是归一化坐标，所以应该是不需要除以p_i(2)的。下面的dep_i应该是不需要的！
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    ////p_i_comp也许可以通过下面标注的方式来求解？但是这个式子本身成立吗？p_i是归一化坐标，引入位姿进行循环转换有什么意义？？
    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i; 式子（1）
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    //p_i_comp如果是通过上面式子（1）的方式求得的，由于第三维z不一定为1，需要除以第三维？
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}