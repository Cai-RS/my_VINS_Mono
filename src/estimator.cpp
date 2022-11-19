#include "estimator.h"

#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_reprojection.h"
#include "backend/edge_imu.h"
//先验边不被创建，直接在代码中构建相关变量

#include <ostream>
#include <fstream>

using namespace myslam;

//用大括号对类对象成员进行初始化（参数化列表），可以避免歧义。当然如果调用的是有参构造函数，则用小括号也是可以的，不会有歧义
//（当构造函数是无参的时候，如果还是用小括号初始化则会产生歧义，则会被误认为是函数定义。但实际上如果有无参构造函数，是不需要在变量后面加小括号的，默认用构造函数初始化）
// https://zhuanlan.zhihu.com/p/268894227
Estimator::Estimator() : f_manager{Rs}
{
    // ROS_INFO("init begins");

    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        //指针赋值为空
        pre_integrations[i] = nullptr;
    }
    for(auto &it: all_image_frame)
    {
        it.second.pre_integration = nullptr;
    }
    tmp_pre_integration = nullptr;
    
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        // cout << "1 Estimator::setParameter tic: " << tic[i].transpose()
        //     << " ric: " << ric[i] << endl;
    }
    cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    f_manager.setRic(ric);
    //投影误差（属于几何误差）的信息矩阵
    project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    //这个指的是什么？
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        //如果pre_integrations[i]确实指向了某个IntegrationBase类对象
        if (pre_integrations[i] != nullptr)
            //虽然delete会释放掉指针指向的内存，但是并不会删除指针。因此在delete指针后，需要再次给指针分配地址（这里让它变为空指针）
            //如果delete后不重新赋值而直接使用该指针，该指针还是会指向原来的地址，也能访问，但是这时原来的地址处存的东西已经不见了，或者很可能已经存入了别的东西！
            //但是可以delete非new出来的内存吗？？其实不是的，从后面的代码可以看出，pre_integrations[i]是指向new出来的内存的！h文件中的pre_integrations只是定义，并没有赋值，例如char* p = new char[5]
            // https://www.kancloud.cn/wangshubo1989/pit/100962
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    //solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    
    tmp_pre_integration = nullptr;
    
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

//dt是两个imu时刻的时间间隔
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    //如果此imu是一个预积分（两帧之间）的初始时刻，则设置预积分的初始时刻imu数据。往后就不需要再这么设置了，因为下一个预积分增量的初始时刻数据就是这个时刻的imu，参考此函数最后两行代码
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    //如果pre_integrations[frame_count]是空（指针），即为0（NULL）。当时间来到新的一帧相机图像，才会创建并初始化新的一个预积分，否则是增量更新当前的预积分量
    //那么从程序上来说，系统开启之后，应该是先执行processIMU，这时frame_count=0，才能创建pre_integrations[0]这个对象（没有用到的变量），然后才执行processImage()
    //这个创建和初始化的命令仅在视觉初始化阶段才会执行，因为之后滑窗阶段的预积分创建命令会在processImage()中，见slideWindow()加*行
    if (!pre_integrations[frame_count])
    {
        // new IntegrationBase{...}，{}及其中内容表示调用构造函数并初始化。如果没有参数，也可以用()，加括号时调用没有参数的构造函数，不加括号调用默认构造函数或唯一的构造函数
        //初始化时需要给定预积分初始时刻的imu数据，这里创建预积分量时就是将第一个和第二个imu时刻设为相同（因为第一个时刻的不可知）
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    //如果不是初始帧，则执行预积分的计算。那么初始帧处的预积分量 pre_integrations[0]（即第一帧到第二帧之间的预积分）该怎么办？只定义不计算吗?从后面的代码来看，是的
    //当系统得到第一帧图像之后，就会立即执行frame_count++，即此时frame_count=1，然后在第二帧图像到来之前，都是在更新pre_integrations[1]
    if (frame_count != 0)
    {
        //integration_base类中有个push_back函数，用于插入新的imu数据并且计算出预积分
        //成员函数push_back()的实现中有propagate()，即预积分的增量更新
        //这里imu的时间戳，应该要大于等于frame_count帧所对应的时间戳。即两帧之间的预积分量是保存在后一帧的序号中的
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        //tmp_pre_integration在processImage()这个函数中进行初始化
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        //这个dt_buf的长度是只有WINDOW_SIZE + 1。dt是两个imu时刻之间的时间差，两帧之间应该有多个dt，所以每个元素用vector
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        //由此可以看出，Rs表示body坐标系到世界坐标系的旋转变换，两帧之间bias假设不变，等于前帧时刻的Bias
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        //中点法
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//图像来一帧就插入all_image_frame一帧，用于追踪特征点。如果是在初始化周期内，就每帧都保留，只插入数据和初始化预积分量。如果是初始化结束，则每来一帧仍插入和追踪，但是会marg（也就是开始滑窗操作）
//image是最新帧图像，带有观测点的信息（通过光流和重新检测得到的）。map中第一个int是特征点id，第二个int不知是啥（帧id或者是相机id）
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    //ROS_DEBUG("new image coming ------------------------------------------");
    // cout << "Adding feature points: " << image.size()<<endl;
    //此函数计算次新帧和次次新帧的视差，如果视差够大，返回true.
    //这里应该不需要担心frame_count等于0，在此函数里可以处理（如果是初始的两帧，则直接进行添加。只有大于等于第三帧时，才利用前面两帧计算视差，决定marg哪一帧）
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;
    //枚举量MARGIN_OLD的值为0，MARGIN_SECOND_NEW值为1
    //ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    //ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    //ROS_DEBUG("Solving %d", frame_count);
    // cout << "number of feature: " << f_manager.getFeatureCount()<<endl;

    //从下面ImageFrame的构造函数可知，header是时间戳
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    //由于tmp_pre_integration是new出来的指针，因此imageframe中的pre_integration也是，因此在使用完之后pre_integration需要delete
    imageframe.pre_integration = tmp_pre_integration;
    //all_image_frame保存的是从系统（重启）运行开始的所有的帧！但是在开始slidewindow时会去除多余的帧，使得其维度不大于WINDOE_SIZE+1
    all_image_frame.insert(make_pair(header, imageframe));
    //这里的acc_0和gyr_0是在processImu()函数中初始化之后的值，因此每个帧时刻，应该是运行processImu()，然后紧接着运行processImage()一次，然后再多次运行processImu()。
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //相机和imu外参需要估计的变量数，如果此值为2，说明没有先验，旋转和位移都要标定
    if (ESTIMATE_EXTRINSIC == 2)
    {
        cout << "calibrating extrinsic param, rotation movement is needed" << endl;
        //非首帧才进行外参的估计，因为非首帧才能得到和上一帧之间的预积分
        if (frame_count != 0)
        {
            //此函数用于找出在frame_count_l和frame_count_r两帧中都被观察到的点，并且返回所有的点对
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            //估计外参数，输入参数为两帧之间的点对，两帧之间的预积分。此处再次说明，两帧之间的预积分量，是保存在后一帧中的序号中的
            //根据CalibrationExRotation函数的实现，外参的估计一定要利用到滑窗内所有帧中的前后点对，即构造的最小二乘系数矩阵要尽可能大地超定。且还要得出的结果满足一定精确度，才会返回结果到变量calib_ric中
            //得到的calib_ric是相机坐标系转到imu坐标系的旋转
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                            //    << calib_ric);
                //ric数组的长度为相机个数，RIC是vector
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                //当第一个滑窗完成了外参的估计后，之后就不再需要再标定了，而是把第一次的估计作为先验进行优化估计
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
    //第一个滑窗周期内都是INITIAL，直到下面完成VIO初始化才改变solver_flag的值。另外，进行下面的初始化之前，一定先完成了上面的外参标定，否则继续滑窗
    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            //第一个条件指初始滑窗成功进行了相机和imu外参的估计，只有这样该值才不等于2。也可以不进行上面的外参估计，而是系统初始化时ESTIMATE_EXTRINSIC就等于1（给定优化初始值）或者0（固定了外参数值，不优化）
            //第二个条件指的是输入两帧图像之间的时间间隔大于0.1，即输入频率小于10？
            if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
            {
                // cout << "1 initialStructure" << endl;
                //整个滑窗所有帧进行初始SFM，估计出各帧imu相对世界坐标系的位姿和速度，三角化满足追踪条件的特征点
                result = initialStructure();
                initial_timestamp = header;
            }
            if (result)
            {
                solver_flag = NON_LINEAR;
                //解决里程计问题，功能是对还没三角化且满足追踪条件的路标点进行三角化，然后进行后端BA优化，并且marg掉某一帧
                //其实刚完成初始化之后是不需要再对特征点进行三角化的，因为在initialStructure()最后已经三角化过了，这里再次计算，结果也不会变
                //之所以要有三角化的操作，是针对初始化成功并marg之后，再进来新的一帧时直接调用solveOdometry()来三角化次新帧到此最新帧中的点
                solveOdometry();
                //完成了初始化之后和marg之后，对滑窗内待估计变量数组Ps,Rs,Vs,Bas,Bgs和f_manager等进行更新维护（去掉marg变量的信息）
                slideWindow();
                f_manager.removeFailures();
                cout << "Initialization finish!" << endl;
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            //如果不满足上上个if，要么是因为前面外参标定没成功，要么是最新两帧间隔时间太短。如果是上个if没满足，则是因为视差不够，sfm不成功。
            //继续滑窗（marg掉某一帧）。至于marg掉哪一帧，需要综合 processImage()开始的设置 和 initialStructure()函数中的执行情况 来判断
            else
                slideWindow();
        }
        else
            //如果不是第WINDOW_SIZE+1帧，则继续接收图像，继续初始化模块；如果是，则frame_count不再改变，之后的每一帧都执行下面的else内的操作
            frame_count++;
    }
    //如果solver_flag 不是INITIAL，则初始化已完成，之后进来的每一帧都只是进行（最新帧的）特征点的三角化和后端优化
    else
    {
        TicToc t_solve;
        //这个函数既进行全局BA，又进行marg （都在函数backendOptimization()中）
        solveOdometry();
        //ROS_DEBUG("solver costs: %fms", t_solve.toc());

        //如果此次solve失败，则重启系统
        if (failureDetection())
        {
            // ROS_WARN("failure detection!");
            failure_occur = 1; //感觉这里的赋值是多余的呀？下一句的clearState()中又立马重置failure_occur=0了.....
            clearState();
            setParameter();
            // ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        //这里滑窗函数内并不进行实际的marg操作，而是在marg后对状态变量数组进行更新维护
        slideWindow();
        f_manager.removeFailures();
        //ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS，这个是VINS的输出
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        //保存这个参数的作用是什么？
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility 检查imu的可观性（加速度应该足够明显）
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        //从第2帧开始执行，因为两帧之间的预积分量，是保存在后面一帧的相对序号中的
        //左增和右增的区别： https://blog.csdn.net/oyhb_1992/article/details/78168458   区别就是一个（左增）可以被赋值，一个不可以
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            //两帧之间的时间差
            double dt = frame_it->second.pre_integration->sum_dt;
            //求帧间平均加速度的累加？
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        //系统运行期间的平均加速度
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        //加速度的标准差？？
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            ////IMU激励不够（近似匀速运动）
             //ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    //Q和T都是有WINDOW_SIZE+1个元素,因为frame_count作为数组索引，最大值就是WINDOW_SIZE（实际上就是数组的第WINDOW_SIZE+1个元素）。但这里是给定数组的维度
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    //用于保存后面sfm过程中被成功三角化的点
    map<int, Vector3d> sfm_tracked_points;
    //sfm_f用于保存所有路标点SFMFeature的所有信息（在所有帧上的被观察值，用tmp_feature这个变量来获取），一个SFMFeature表示一个路标点。那为什么不直接使用f_manager（它也是所有路标点的信息，也有估计的点的深度）
    //sfm_f这个变量只是临时变量，用来进行视觉的初始化
    vector<SFMFeature> sfm_f;
    //feature的类型是list<FeaturePerId>，FeaturePerId表示一个路标点的所有信息，即该点在它被观察到的所有帧中的信息（包括在被观测初始帧中的估计深度）
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        //表示所有的点都还未被三角化
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        //feature_per_frame的类型为vector<FeaturePerFrame>，FeaturePerFrame是 某id的点在某帧中的“观测”信息，包括三维归一化坐标，二维像素坐标和像素速度
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            //point应该是三维归一化坐标，因为后面的observation是二维坐标，两者能部分赋值，只可能都是归一化坐标？
            Vector3d pts_j = it_per_frame.point;
            //某个路标点的所有被观测值（应该是归一化坐标，省略了保存z=1)
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //这个判断啥意思？find previous frame which contians enough correspondance and parallex with newest frame。这个函数只能在最新帧为滑窗内第WINDOW_SIZE+1帧才可以，也就是说完成初始化之后
    //求得的relative_R使得最新帧旋转到第l帧。但是这个函数里没有把这两帧中三角化的点给保存进sfm_f，这两帧之间的点三角化在函数construct()里
    if (!relativePose(relative_R, relative_T, l))
    {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }
    GlobalSFM sfm;
    //Q，T，sfm_f和sfm_tracked_points是被求解的参数，Q表示从相机坐标系转到参考坐标系（第l帧相机坐标系），relative_R表示最新帧坐标系旋转到参考坐标系
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        cout << "global SFM failed!" << endl;
        //为什么全局sfm失败，就要marg掉最老的帧？为什么不是marg掉次新帧？从construct()函数可以看出，sfm失败最大的可能还是因为第l帧和初始帧的共视点太少了
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame  上面sfm.construct只是针对关键帧进行了优化，接下来是要对all_image_frame保存的所有帧进行PnP计算
    //all_image_frame变量并不总是只保存滑窗内的变量，因为当marg次新帧后并不会去除all_image_frame中的相应帧信息，只会在marg最旧帧后一次性去掉滑窗最旧帧及其之前的帧，见slideWindow()函数
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        //意思是Headers只保留滑窗内关键帧的时间戳
        //如果all_image_frame中的此帧为关键帧，则不需要对它sfm了，因为它在前面construct函数已经优化了，只需要把估计出的参数输入all_image_frame对应帧中，然后执行下次循环检查下一帧
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            //转化成第i帧的imu坐标系到世界坐标系（也是第l帧相机坐标系）。RIC表示从相机坐标系转到imu坐标系
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            //位移这里先不需要转变，就是从第i帧的相机坐标系到世界坐标系
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        //当上一帧是关键帧时，i已经加1了，为什么这里会出现下一帧的时间戳比下一帧关键帧大的情况？？关键帧应该比较稀疏啊？？
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        //i加1，使得这个未被估计的帧的位姿取得更合理的初始估计值
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        //points的类型为map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
        for (auto &id_pts : frame_it->second.points)
        {
            //points中第一个int表示点的id
            int feature_id = id_pts.first;
            //表示各id点在此帧中的观测信息？如果是这样，那vector中应该只有一个元素（一个pair），那为什么要用vector？？第二个int又表示什么？帧id吗？
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    //得到被观测点的世界坐标
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    //得到该（未被估计位姿的）帧中已知世界坐标的观测点
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        //实际上求解PnP最少只需要3点（P3P）就可以了，但要求不共线的三点，这里为了保证精度，要求至少要有6点
        if (pts_3_vector.size() < 6)
        {
            cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << endl;
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            cout << " solve pnp fail!" << endl;
            return false;
        }
        //rvec使坐标点从世界坐标系（也是第l帧的坐标系）转到相机坐标系（即相机坐标系相对于世界坐标系的位姿）
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        //****
        T_pnp = R_pnp * (-T_pnp);
        //最后的R是从第i帧的“imu”坐标系到世界坐标系（第l帧相机坐标系）的旋转，T表示的是从第i帧的“相机”坐标系到世界坐标系（第l帧相机坐标系）的位移
        //但是由于相机和imu之间假设为没有位移，因此T也可以说是从第i帧的“imu”坐标系到世界坐标系（第l帧相机坐标系）的位移
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    //上面完成了视觉的单独初始化（所有帧的sfm），接下来就是视觉和imu的对齐。上面视觉初始化过程中是把第l帧相机坐标系设定为与世界坐标系没有相对位姿，在VIO对齐中需要矫正过来，需要固定的是初始帧的imu坐标系！！
    //VIO对齐的结果，就是计算出各帧imu坐标系相对“世界坐标系”的位姿和速度Ps,Rs和Vs。g在初始imu坐标系下的表示，以及g_bias的初始估计（初始预积分量中的bias均是取0，这里估计后需要更新预积分）
    //为了防止漂移，设定初始帧imu坐标系和世界坐标系 原点相同，没有yaw偏角，即固定了四维以解决VIO不可观问题
    if (visualInitialAlign())
        return true;
    //算出来的Ps、Rs和Vs没有保存进all_image_frame中，即all_image_frame中的R仍然表示视觉初始化后的各帧imu坐标系变化到参考帧l（还没有和imu对齐），T仍表示从各帧相机坐标系变换到参考帧l
    else
    {
        cout << "misalign visual structure with IMU" << endl;
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale  完成了IUM的陀螺仪bias估计和VIO对齐（利用的是上面完成的视觉初始化，认为视觉的估计足够精确）
    //x是线性变量的初始化估计结果（即所有帧时刻对应的imu坐标系的速度（表达在该帧imu坐标系下），表达在参考帧坐标系下的重力向量，尺度因子）
    //g是从x中单独取出来的结果，Bgs是估计出来的陀螺仪Bias
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        //ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    //得到满足追踪成功条件（至少连续两帧，且首帧至少是次次新帧再往前的点集的深度。
    // 之所以要设置观测起始帧在次次新帧之前，是因为
    //下面仅对这些点重新进行三角化（因为从construct()函数开一直都是固定特征点位置，仅优化位姿，这里则需要固定优化后的位姿，仅优化特征点位置）
    //以倒数第三帧为观测初始帧并被连续追踪到最新帧的点，虽然下面没被重新估计，但也已具有估计的深度（在construct()函数中），所以它们最后不会被removeFailures()函数清除
    //getDepthVector()函数是从feature中得到满足条件的点集和它们的逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    //clearDepth()结果是把这些点（在feature中的）的估计深度设为-1
    f_manager.clearDepth(dep);

    //triangulate on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    //为什么把相机和imu的位移直接设为0？？
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    //RIC在视觉初始化时已经标定好了
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    //对“满足追踪成功条件”的点进行重新三角化，此时用的Rs和Ps是还未经过VIO对齐矫正的，那算出来的点深度是对的吗？
    //计算出来的点深度是该点在其“被观测首帧”下的坐标深度，计算过程利用的是其他帧相对这个首帧的相对位姿，不受Ps和Rs的计算参考帧的影响，所以后面VIO对齐后不需要修正
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    //x.tail<1>()本身是一个vector，（x.tail<1>()(0)表示取它的第一个元素
    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        //初始化进程的滑窗内的预积分中，bias是随意给定的，并没有经过优化，这里用VIO对齐时估计出来的陀螺仪Bias来更新。加速计的Bias没有估计，因此给的仍是初始值
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        //****
        //Ps[i]将第i帧imu坐标系移动到初始帧的imu坐标系（即初始帧imu坐标系原点指向i帧imu坐标系原点的向量），“表达在参考帧l坐标系下”！！
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            //将i帧对应的imu速度（表达在imu坐标系下）转换到表达参考帧（第l帧）的坐标系下
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    //除了这些满足追踪成功条件的特征点，别的点不也得恢复尺寸？？别的点不可取，因为要么不是追踪成功的，要么可能会在marg次新帧之后变成无用点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    //g2R()为静态函数
    //R0表示将第l帧相机坐标系的点旋转到世界坐标系
    Matrix3d R0 = Utility::g2R(g);
    //R0 * Rs[0]表示将初始帧imu坐标系下的点旋转到世界坐标系下，取这个旋转中的yaw角
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    //下面这个操作是什么原理？因为要固定初始帧imu相对世界坐标系的yaw角（为0），因此要对其他的坐标系都施加这个yaw角的偏差
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    //把g先从l帧坐标系下表示转到世界坐标系，"然后再转到初始帧imu坐标系下"（后面一步存疑）
    //在视觉初始化里面，由于系统有7维不可观，我们固定了第l帧相对世界坐标系的位姿（6维）和最新帧的相对第l帧的位移。
    //由于VIO中只有四维不可观，所以这里我们选择把初始帧相对世界坐标系的位移固定为0（3维），把初始帧imu相对世界坐标系的yaw角为0(1维），这点从下面的Rs[0] = rot_diff * Rs[0]可以看出，就是把两者相对旋转中的yaw角给消去了！
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    //得到参考帧l到世界坐标系的旋转
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        //得到的是第i帧imu变到初始帧imu的位移，表达在世界坐标系下。
        //Ps前面乘以rot_diff，原因可见代码第450行和第518行（标记了****记号），在前面计算Ps的过程中，每个加项都用到了Rs，Ps的更新来自于Rs的更新，根据第563行Rs的更新式子，因此直接乘上rot_diff即可
        //并且由于设定世界坐标系的原点和初始帧imu坐标系的原点是重合的，因此这里Ps[i]最终表示的是将第i帧imu坐标系移动世界坐标系（表达在世界坐标系下）
        Ps[i] = rot_diff * Ps[i];
        //Rs[i]表示从第i帧imu坐标系旋转到世界坐标系
        //当i=0时，可以看到是设定了世界坐标系与初始帧imu之间没有yaw角的偏差（强制消除，并由于各帧之间相对位姿应该保持不变，因此所有帧imu位姿都要添加这个偏差变换），只有pitch角和roll角的偏差。
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
        //算出来的Ps、Rs和Vs没有保存进all_image_frame中，即all_image_frame中的R和T仍然表示各帧imu坐标系变化到参考帧l
    }
    //ROS_DEBUG_STREAM("g0     " << g.transpose());
    //ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //两帧之间的平均视差需要满足要求，这是经验公式。求得的relative_R使得最新帧旋转到第i帧
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                //如果满足此条件，则认为此帧适合作为初始化sfm的参考帧
                l = i;
                //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        //已知关键帧的位姿，再次三角化所有的特征点
        //经过了227行的initialStructure()函数之后，Rs（f_manager的内部变量）和Ps指的是各帧imu坐标系变换到世界坐标系的位姿（这是正确的，详细分析见initialStructure()函数）
        f_manager.triangulate(Ps, tic, ric);
        //cout << "triangulation costs : " << t_tri.toc() << endl;    
        //进行后端优化
        backendOptimization();
    }
}

//把状态变量的格式变为double型数组，符合ceres的要求
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        //进入优化前，Bas的初始值是0
        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        //进入优化前，Bgs的初始值是在VIO对齐中估计的值
        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    //进入优化前，tic的初始值应该是0（即使是先验值），ric的初始值是外参标定的结果
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    //相机和imu的时间差。此项目中是设置ESTIMATE_TD = 0 ，因此problemSolve()函数中就没考虑需要估计td的情况
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];
    //什么时候会出现failure_occur==1的情况？貌似没看到这种可能性？
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    //优化之后的初始帧imu位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    //貌似origin_R0.x()就是等于0，因为每次估计（包括视觉初始化和VIO对齐步骤）后都会将首帧imu相对世界坐标系的yaw角矫正为0？
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO  矫正位姿，优化后还是要假设初始帧imu坐标系与世界坐标系没有相对yaw角，对所有帧的Rs[i]左乘这个矫正旋转矩阵，整体调整位姿
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    //如果出现了欧拉角奇异值，即某位姿对应的俯仰角为±90度时，从旋转矩阵R反推偏航和滚转角时可以有多种解。此时求解yaw角差值就不再适合用上面的单独求ypr中的yaw角了
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        //ROS_DEBUG("euler singular point!");
        //常用的四元数格式有Quaternionf(float)和Quaterniond(double)
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        //要注意的是，只有单位四元数才表示旋转矩阵，所以要先对四元数做单位化。
        // //也就是说，世界坐标系的
        //但如果出现欧拉角奇异值的话，Rs[0]前后就不会有什么改变了（因为此时ypr三个角均被修正了），但是其他的Rs[i]仍可能改变，因为各帧之间的"相对位姿"在优化过程中会被改变
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }
    //对点深度也进行了优化估计
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        //prev_relo_t貌似没被初始化过？
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        //闭环的两帧之间的相对位姿
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
    }
}

bool Estimator::failureDetection()
{
    //last_track_num表示最新帧图像中中追踪到的点数
    if (f_manager.last_track_num < 2)
    {
        //ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    //为什么只检查最新帧对应的bias？？
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        //ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        //ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        //ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    //此轮优化后的最后一帧与上一轮优化的最后一帧之间的位移。但是为什么不直接用这轮优化的最后一帧到次新帧的位移？？
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        return true;
    }
    //沿z轴的位移太大，意味着两帧之间的视差也会差别很大（视差至于点的深度差距有关）
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        //ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::MargOldFrame()
{
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }
    //把所有关键帧的VertexPose和VertexSpeedBias按时间顺序加入problem
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }
    //为什么problem里没有添加VertexInverseDepth类的点？？添加了，在下面Visual Factor部分的代码中，只添加和首帧相关的特征点

    // 需要marg的IMU边
    {
        //初始帧和第一帧之间的预积分。相机两帧之间的时间差小于10ms？
        //为什么只用首个预积分？因为这里是marg掉最老的帧，只有初始预积分与其相关
        if (pre_integrations[1]->sum_dt < 10.0)
        {
            std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[1]));
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(vertexCams_vec[0]);
            edge_vertex.push_back(vertexVB_vec[0]);
            edge_vertex.push_back(vertexCams_vec[1]);
            edge_vertex.push_back(vertexVB_vec[1]);
            imuEdge->SetVertex(edge_vertex);
            problem.AddEdge(imuEdge);
        }
    }

    // Visual Factor 需要marg的视觉观测边
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            //只找能被初始帧观测到的特征点
            if (imu_i != 0)
                continue;
            //归一化坐标
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX inv_d(1);
            inv_d << para_Feature[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            //在此problem只添加和初始帧相关的特征点，无关的特征点不添加
            problem.AddVertex(verterxPoint);

            // 遍历该点除在首帧之外的所有观测
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                //首帧不计入
                if (imu_i == imu_j)
                    continue;
                //归一化坐标
                Vector3d pts_j = it_per_frame.point;
                //和 观测首帧位姿 相关的边都要被marg。视觉重投影误差是用每个点的观测初始帧与其他观测帧
                std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                //VertexInverseDepth类的子类点
                edge_vertex.push_back(verterxPoint);
                //VertexPose类的子类点
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                //VertexPose类的子类点
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                // project_sqrt_info_为固定值，已给定
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);
                //给定计算损失函数的方式，输入LossFunction类指针
                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            // 但是这个 prior 还是之前的维度，需要扩展下装新的pose.新的一帧的状态变量为15维，PRV和两个bias
            problem.ExtendHessiansPriorSize(15);
        }
        else
        {
            //没有先验，也要设置这些参数
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
        }
    }

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    marg_vertex.push_back(vertexCams_vec[0]);
    marg_vertex.push_back(vertexVB_vec[0]);
    //只marg帧位姿，不marg帧内的观测点？观测点也需要marg，这在Marginalize()函数内实现
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_ = problem.GetHessianPrior();
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}

void Estimator::MargNewFrame()
{

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    //    vector<backend::Point3d> points;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);

            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
        }
    }

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    // 把窗口倒数第二个帧 marg 掉
    marg_vertex.push_back(vertexCams_vec[WINDOW_SIZE - 1]);
    marg_vertex.push_back(vertexVB_vec[WINDOW_SIZE - 1]);
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_ = problem.GetHessianPrior();
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}

void Estimator::problemSolve()
{
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);
    //    lossfunction = new backend::TukeyLoss(1.0);

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);

        if (!ESTIMATE_EXTRINSIC)
        {
            //ROS_DEBUG("fix extinsic param");
            // TODO:: set Hessian prior to zero
            vertexExt->SetFixed();
        }
        else{
            //ROS_DEBUG("estimate extinsic param");
        }
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // IMU  这里是全局BA，因此需要用到滑窗内所有的预积分量
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        //pre_integrations[0]是不算的，因为首帧和第二帧之间的预积分保存在pre_integrations[1]中
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;

        std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[j]));
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        edge_vertex.push_back(vertexCams_vec[i]);
        edge_vertex.push_back(vertexVB_vec[i]);
        edge_vertex.push_back(vertexCams_vec[j]);
        edge_vertex.push_back(vertexVB_vec[j]);
        imuEdge->SetVertex(edge_vertex);
        problem.AddEdge(imuEdge);
    }

    // Visual Factor  需要所有（满足追踪条件）的特征点从其被观测首帧到其他所有被观测帧的重投影误差边
    vector<shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX inv_d(1);
            inv_d << para_Feature[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);
            vertexPt_vec.push_back(verterxPoint);

            // 遍历所有的观测
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Vector3d pts_j = it_per_frame.point;

                std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            // 外参数先验设置为 0. TODO:: 这个应该放到 solver 里去弄 （H中与外参数有关的部分为0，因为H对角线某子块为0，说明其雅可比为0）
            //            Hprior_.block(0,0,6,Hprior_.cols()).setZero();
            //            Hprior_.block(0,0,Hprior_.rows(),6).setZero();

            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
    }

    //但VINS选择忽略FEJ策略（认为加入了反而误差大），即先验中的线性化点和其他误差成分中的线性化点不相同
    //（FEJ策略认为相关点在其他误差中的线性化点应该始终和先验中的相同）
    problem.Solve(10);

    // update bprior_,  Hprior_ do not need update
    //Solve()函数里的UpdateStates()函数会更新b_prior和相应的err_prior_，而J_prior_inv_不需要更新（FEJ策略）
    if (Hprior_.rows() > 0)
    {
        std::cout << "----------- update bprior -------------\n";
        std::cout << "             before: " << bprior_.norm() << std::endl;
        std::cout << "                     " << errprior_.norm() << std::endl;
        bprior_ = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        std::cout << "             after: " << bprior_.norm() << std::endl;
        std::cout << "                    " << errprior_.norm() << std::endl;
    }

    // update parameter
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        VecX p = vertexCams_vec[i]->Parameters();
        for (int j = 0; j < 7; ++j)
        {
            para_Pose[i][j] = p[j];
        }

        VecX vb = vertexVB_vec[i]->Parameters();
        for (int j = 0; j < 9; ++j)
        {
            para_SpeedBias[i][j] = vb[j];
        }
    }

    // 遍历每一个特征
    for (int i = 0; i < vertexPt_vec.size(); ++i)
    {
        VecX f = vertexPt_vec[i]->Parameters();
        para_Feature[i][0] = f[0];
    }
}

void Estimator::backendOptimization()
{
    TicToc t_solver;
    // 借助 vins 框架，维护变量
    vector2double();
    // 构建求解器  求解器中随着优化有迭代更新para_Pose等变量
    problemSolve();
    // 优化后的变量处理下自由度（4个自由度需固定）
    double2vector();
    //ROS_INFO("whole time for solver: %f", t_solver.toc());

    // 维护 marg
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        vector2double();
        //在marg之后para_Pose等变量中元素的排序并没有变动（即没有把marg掉的元素去除）
        MargOldFrame();

        std::unordered_map<long, double *> addr_shift; // prior 中对应的保留下来的参数地址
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            //把每个指针的地址作为哈希函数的索引（key），几乎具有唯一性 （addr_shift中只有WINDOW_SIZE各元素，因为marg掉了一帧）
            //只使用“保留下来的变量原来的地址”作为索引，但是为什么marg了最老帧，这里还要保留para_Pose[0]？？？
            //reinterpret_cast运算符是用来处理无关类型之间的转换；它会产生一个新的值，这个值会有与原始参数（expressoin）有完全相同的比特位
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
    }
    else
    {
        if (Hprior_.rows() > 0)
        {

            vector2double();

            MargNewFrame();

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    //为什么要保留para_Pose[WINDOW_SIZE - 1]，即次新帧的位姿信息？
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
        }
    }
    
}


void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);
                //这个是标准库里面的函数，不是vector类中的成员函数
                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            //去点最老帧之后，数组的第WINDOW_SIZE+1个元素和第WINDOW_SIZE个相同，都等于最新帧的数据？
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            //即将进来的下一帧的bias初始化为当前最新帧优化出来的bias
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            //为当前最新帧和即将进来的下一帧之间的预积分创建对象并初始化
            //**先执行processIMU，再执行processImage，因此此时acc_0和gyr_0等于新一帧到来时刻的加速度计和陀螺仪的数据，正好用于初始化新的预积分变量
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            //始终执行
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                //释放该帧中的预积分指针的占用内存，并且赋为空指针。pre_integration被初始化时被赋予了new出来的指针，需要delete
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
                //从整个系统开始运行后的初始帧到此刻被marg的最老帧，所有帧中的预积分指针释放内存并赋为空指针
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }
                //从初始帧到此刻最老帧，都去掉（第一行中不包括it_0这个迭代器，因此需要单独再去掉它）
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            //dt_buf[frame_count]是个vector，记录某两帧图像之间的所有imu时刻
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                //两帧之间的各时刻的imu信息，也都是保存在后一帧对应的数组位置。那也dt_buf[0][]是没用的信息，跟pre_integrations[0]一样
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
                //对倒数第三帧到倒数第二帧（次新帧）之间的预积分继续执行增量更新（即把 次新帧到最新帧之间的增量部分 添加到这里）
                //目的是使去掉次新帧的信息之后，预积分信息能够连贯
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            //清除第WINDOW_SIZE+1个元素中的内容，以便存入下一帧开始的imu和预积分信息
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            //all_image_frame变量并不总是只保存滑窗内的变量，因为当marg次新帧后并不会去除all_image_frame中的相应帧信息
            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    //这个量指marg次新帧的操作次数。有什么用？貌似系统没用到
    sum_of_front++;
    //对feature中的观测帧进行删减，并且去除仅被次新帧观测到的点
    f_manager.removeFront(frame_count);
}


// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    //这个量指marg最旧帧的操作次数。有什么用？貌似系统没用到
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        //back_R0是没有进行移位操作之前（即还没从Rs中去除首帧的信息）的Rs[0]，得到的R0是移位之前首帧相机坐标系到世界坐标系的旋转，P0同
        R0 = back_R0 * ric[0];
        //下面的Rs[0]则是没有移位操作之前的Rs[1]
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        //这种情况发生在VIO初始化期间，当初始化不成功时，直接去掉最旧帧的信息，并且把其他特征点的起始帧-1，下一帧进来后继续VIO初始化
        f_manager.removeBack();
}
