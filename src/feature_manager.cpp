#include "feature_manager.h"

//�����������֡����ţ�ǰ����֡�������ģ�����ĸ��ٲ��ܶ�
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

//���б�����е����ݽ������
void FeatureManager::clearState()
{
    feature.clear();
}

//�������ɹ�׷�ٵĵ�id����Ӧ����ָ�����ڵģ�
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();
        //it.start_frame < WINDOW_SIZE - 2 ����Ŀ����ʲô����˵�����ʼ֡���������������ڼ��ǰWINDOW_SIZE - 2֮֡�ڣ��������������ô������ν�����ڼ����ÿ��������
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

//�˺�����������֡�еĵ����Ϣ��ӣ���feature�У������Ҽ���Ӳ�
//frame_count��֡����ţ������ǻ����еĵ�һ֡������Ӧ�ò��õ����ǵ�һ֡����Ϊǰ�漸֡���ڳ�ʼ���Ľ��̣���map�е�һ��intӦ���ǵ��id���ڶ���int�����id��һ��map�п��Դ洢����Ԫ�أ���vectorһ��
//�����ر�����tureʱ�����������ڶ�֡��������֡Ӧ�ñ���������Ϊ��������֮֡����Ӳ��㹻��
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    //ROS_DEBUG("input feature: %d", (int)image.size());
    //ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts : image)
    {
        //����Ϣ��ʱ���������Ϊʲôʼ����second[0]�����ⲻ����Զȡvector�еĵ�һ��Ԫ���𣿿�������Ϊֻ��һ��Ԫ�أ�
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        //����STL�ֲ��е�find_if�������壬����������Ϊ�Զ����predicate function
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
        //�����find_if�����У�����Ҳ�������������it���������ص��ǵڶ�����������feature.end()
        //ע�⣬feature.end()���������ָ��λ����list�����һ��Ԫ��λ���ټ�1��
        if (it == feature.end())
        {
            //feature��list������ĵ�id�ǰ��ձ��۲��ʱ��˳�򱻷Ž��ģ�����Ԫ�ؾ������µ�֡�б��۲⵽���µ�
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }
    //Ҫô��֡Ϊͷ��֡��Ҫô��֡��׷�ٵ��ĵ���С��20��׷�ٵ��ĵ����٣�˵�����µ���֮֡���Ӳ�󣬲���Ҫ�������Ӳ���㣩
    if (frame_count < 2 || last_track_num < 20)
        return true;
    //�����if�Ǵ�����֡�ʹ���֡��������Ĺ���������last_track_num������������֡�Ƿ�ؼ�������޷��жϣ���Ӵ���֡�ʹδ���֡���Ӳ����ж�
    for (auto &it_per_id : feature)
    {
        //��һ��������ʾ��ʼ֡С�ڻ���ڵ�������֡
        //�ڶ���������ʾ���������ӱ��۲⵽��ʼ���������ڶ�֡�����ںţ����ߵ�����һ֡�����ںţ�һֱ���������ع۲⵽
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            //compensated�г��ģ��˺������㵹���ڶ�֡�͵�������֮֡����Ӳ�
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
    //parallax_num == 0����������ܳ����𣿣��п��ܣ������Ļ���������֡��֮ǰ��֡�Ͷ��ᱻ�𽥵�marg������Ϊ���ǵĵ�ӵ����ڶ�֡���Ѿ�û���������ˣ�
    //��ʱ����֡�ʹδ���֮֡��û�й�������ȻҪ��������֡
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        //�������֡������֮֡���Ӳ����last_track_num����С�����Ҵ���֡�ʹδ���֡���Ӳ�Ҳ����������Բ���������֡
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

//�˺��������ҳ���frame_count_l��frame_count_r��֡�ж����۲쵽�ĵ㣬���ҷ������еĵ��
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
 //�������������x��Ӧ��������ȣ���Ϊestimated_depth�����
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //ֻ����˳ɹ�׷���ҷ���Ҫ��ĵ����ȣ��������ʱҲ�����������ɸѡ�͸�ֵ
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

//ֻ����Щ�Ѿ������ù���ȣ�setDepth�����������������ȻΪ���ģ���ʼ��ʱ����Ⱦ��Ǹ��ģ���Żᱻ�Ƴ�
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

//��setDepth������������ʲô��
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

//���������д��Vector3d Ps����Psֻ��һ����ά������������������[]�����Ϊ��vector<Vtctor3d>��
//�ӵ�������������������������������������imu��λ�Ʋ���ticΪ0
//�����������ǻ���ԭ���VIO�γ̡��Ӿ�ǰ�ˡ�һ�ڵĿμ���ע������Ĵ���ĵ�3D����������۲���֡�µ�
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //ֻѡ������׷�ٳɹ������ĵ����������ǻ�
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //�����ȴ���0��˵���Ѿ������ƹ��ˣ�
        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        //�����洢ÿ������������ǻ���������ʽ��ÿ����ʽ�Ľ����1*4ά����λ�˾����һ��
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        //tic[0]��vector<Vector3d>�еĵ�һ��Ԫ�أ���ʵ�������vector�����Ҳ��һ��Ԫ��
        //Ps[i]ָ���ǵ�i֡ʱ�̵ġ��������ϵ�����ο�֡l���������ϵ��������������ϵ����λ�ƣ�����ڲο�֡l����������ϵ�£�,
        //�������������imu֮�����Ϊû��λ�ƣ�tic=0)�����PsҲ����˵�Ǵӵ�i֡�ġ�imu������ϵ����������ϵ����l֡�������ϵ����λ�ơ����tic������0����ֱ����t0=Ps[imu_i]�ͺã�����t0���ܱ�ʾʹ��i֡����Ƶ���������ϵ
        //Rsָ���Ǹ�֡ʱ�̵ġ�imu����ϵ�����ο�֡l���������ϵ��������������ϵ������ת��ric���������imu����ת
        //������forѭ���еĴ���������PsӦ����ĳ֡�����Ӧ��imu����ϵ����ʼ֡������������ϵ����λ�ƣ�Rs���ƣ�ric�������imu����ת��
        //Rs��FeatureManager���еĳ�Ա������Ҫ�Ӳ�������
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();
        //����õ㱻�۲쵽������֡��Ըõ�۲��ʼ֡��λ�ˣ��γɲ�������P
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            //�Ǵ�imu_j = imu_i��ʼ�ģ�˵���۲��ʼ֡�еĹ۲�Ҳ�ܱ��������ǻ�����ʹҪ�����Ǹõ��ڹ۲��ʼ֡�µ�����
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            //r0ָ��r1�������R0�У��������֡j�任���õ㱻�۲쵽�ĳ�ʼ֡����������λ���ǲ��ܲο�֡lѡ���¼��������Ps��Rs��Ӱ���
            //��˺����������ĵ����Ⱦ��Ǳ���ڸõ㱻�۲쵽�ĳ�ʼ֡����ϵ�µ���ȣ�VIO�Ķ�����������ֵû��Ӱ��
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            //����Ĺ�һ������ָZ����Ϊ1������������ά�����ģΪ1����������P.row(0)��P.row(1)ǰ��Ҫ����f[2]
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
            //Ϊʲô������Ƿ��������棿��Ϊ�۲��ʼ֡�еĹ۲�Ҳ���ṩ��Ч����������
            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        //������������ά����ε����꣬��Ϊ���һλ����1�����Ż�ʱ�����Լ����Ϊ����������ģ��Ϊ1
        //���������svd�ķ�����⣬����С����ֵ��Ӧ��������������Ϊ�Ż��������Ȼ���ðѵ���ά��һ
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];
        //�õ������ȣ����������ڳ�ʼ֡�µ�
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        //������Ƴ��������̫С������ΪĬ����ȣ�ֵΪ5.0����ΪʲôҪ��ô����
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
        //�������i��ɶ�ã���
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

//�˺����ǰѻ��������֡��marg֮����Ҫ��feature_manager�е���������Ϣ����ɾ������������
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //�������ڵĵ�һ֡���Ϊ0
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            //ѡ����ʼ֡�е������㣬��Ϊestimated_depth������ʼ֡�е�����
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            //����õ�������֡�ʹξ�֡�б���������ֱ��ȥ����������
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                //���۲���֡�������Ϣ�����֡ת�Ƶ��ξ�֡��
                //estimated_depthΪ��ȣ���ôuv_iӦ���ǹ�һ������
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                //�õ�����������ϵ�µ�����
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                //�õ��ڵ�j֡���ξ�֡���������ϵ�µ����꣬new_PΪ�������ϵjָ����������ϵ���������������ϵ�£�
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
        * //WINDOW_SIZE-1����Ϊ��0��ʼ����
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}
//�˺����ĵ���ʱ���ǳ�ʼ��ʧ�ܣ�������Ҫmarg�����֡ʱ����ʱ��������sfm���ƣ�û��marg��ֱ��ȥ�����֡��Ϣ���ɣ���һ֡���������VIO��ʼ��
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

//�����frame_countָ������֡������Ĵ���֡�ǵڼ�֡������ʵ��idӦ��Ϊframe_count-1�������marg��������֡�Ѿ�����feature֮���ٽ��е�
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        
        //if����������Ϊ�棬˵��������������ʼ֡��Ҫ��marg��֡�ĺ�һ֡����������ֻ��Ҫ������һ����
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            //������֡�۲⵽�ˣ���ȥ������֡�еĹ۲���Ϣ
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            //������������˵�������������ڱ�marg�Ĵ���֡�б��۲⵽��������֡��Ҳû����
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame���Եģ�
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;
    //Ϊʲô�����ڶ�֡�е�point�Ͳ���Ҫ������ȣ���point����Ӧ�þ��ǹ�һ�����꣬����Ӧ���ǲ���Ҫ����p_i(2)�ġ������dep_iӦ���ǲ���Ҫ�ģ�
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    ////p_i_compҲ�����ͨ�������ע�ķ�ʽ����⣿�������ʽ�ӱ��������p_i�ǹ�һ�����꣬����λ�˽���ѭ��ת����ʲô���壿��
    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i; ʽ�ӣ�1��
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    //p_i_comp�����ͨ������ʽ�ӣ�1���ķ�ʽ��õģ����ڵ���άz��һ��Ϊ1����Ҫ���Ե���ά��
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}