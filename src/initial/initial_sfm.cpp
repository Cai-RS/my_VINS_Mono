#include "initial/initial_sfm.h"

GlobalSFM::GlobalSFM(){}
//����Ϊ���캯���Ķ���

//�˺���Ϊ��֪ĳ����������֡�Ĺ۲⣬��������ķ������������ά�����磩���꣬��Ϊֻ�����ι۲⣬�պ����ȣ��ǳ���������Ҫ��С���ˣ�
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	//ͬһ�����ڲ�ͬ֡Pose0��Pose1��ָ������������ϵ���߲ο�֡����ϵ����֡��ת�������µĹ۲�ֵpoint0��point1����Ҫ�����ǻ����õ�����������ϵ���ߵ�һ֡�е����꣨��һ����4ά��
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	//�������SVD�ֽ⣬�ֽ�õ������������������о��Ǵ����������������ֵ��С��Ӧ�������������������ǽ������ĵ��ĸ�����Ϊ1
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}



//�������iӦ��ָ���ǵ�i֡ͼ�񣬴˺���Ϊ��֪�������ڵ�i֡�еĹ۲���������磨��ο�֡������ϵ�µ���ά���꣬�����Ż���ʼֵ����BA���Ż����λ�ˣ����ﲢδ�Ż���ά���꣩
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	//jΪ�����ĸ��������ظ���
	for (int j = 0; j < feature_num; j++)
	{
		//��Ҫ�ҵ���ȣ�����ڲο�����ϵ�£��Ѿ������Ƴ����ĵ�
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		//sfm_f�е�ÿ��Ԫ�ش�����һ����������ֵ�֡�����Ӧ��ά�۲죬��j����ĳ�������㣬k����õ��������Щ֡���������j��kʱΪ�˰ѵ�i֡�й۲쵽�������������Ϣ��ȡ����
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				//ĳ��id�ĵ��ڵ�i֡���ֻ�ᱻ�۲⵽һ�Σ��������ҵ��˹۲⣬ֱ���˳�ѭ��
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	//KӦ����������ڲξ���DӦ�����������������ֵ��revc�ǳ�ֵ��ת����ת��Ϊcv�ĸ�ʽ���ٱ仯Ϊ��ת�����ĸ�ʽ��tͬ��Ҫ�����Ż�D, rvec, t
	bool pnp_succ;
	//solvePnP��֪����Ϊ���Ƶ� A,B,C�����磨�ο�������ϵ�µ�λ�� �Լ� ��������е�ͶӰ���ꡣPnP����������Ҫ����㣬��P3P
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//��ת����ת��Ϊcv��ʽ����ת����
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}



// �˺���Ϊ��֪ĳ��֡����ԣ���������ʼ֡����ϵ��λ��ʱ����������֡��λ����������֡�й�ͬ�۲������ǻ���������������������ϵ�µ����꣩������Ӧ�����ǻ������ܶ�ĵ㣬�Ա�����֮֡������֡��PnP�ܹ�˳��
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	// �ж�Ϊ��ͬ֡����Ӧ��200�л������֡���ǵ�frame_num�����
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)
	{
		//Ϊʲô������trueʱ���������м��㣿true��ʾ��·������ά���꣨����������ϵ���߳�ʼ֡�£��ѱ���⣬��Ӧ��200�У���Щ�ڵ�i�͵�frame_num-1֡�еĹ�ͬ�۲���Ѿ������ǻ��ˣ�ֻ�����ǻ��µĹ�ͬ��
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}



// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
// �������lΪ��ѡ����ʼ���Ĳο�֡id������ͨ������������֡���������������Ӳ�͹�����ľ�֡��relative_R��relative_T������������֡�����λ��
//frame_num�����ֵ����frame_count+1����frame_count=WINDOW_SIZE�����frame_num=WINDOW_SIZE+1����frame_num��ʾ��������ĳ��ȣ�
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view  �ȶԲο�֡l������֡���г�ʼ��
	// �����q[l]Ϊָ����֡������Ϊʲô��������磨���ʼ֡������ϵû����ת��ƽ���أ�˵�����ͱ�����Ϊ����������ϵ�����λ��Ϊ0������ں����λ���Ż�ʱ�Ĺ̶�λ��Ҫ��Ӧ
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	//q����ĳ���Ϊframe=WINDOW_SIZE+1������������Ԫ������Ϊframe_num - 1��������֡
	//relative_R������֡�������ϵ��ת����������ϵ
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	//��Ԫ������
	Quaterniond c_Quat[frame_num];
	// c_rotationΪ��Ԫ��ά�ȵ����飬����һ��Ϊ��Ԫ�������ֶ���֤������ceres�У���ʽҪ��
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	// �������������������������q��ʾ��ĳ֡�������ϵת���ο�֡�������ϵ����l֡������Ԫ������Ϊ����Ҫ����triangulateTwoFrames��λ�˲���Pose���ǴӲο�����ϵת���������ϵ
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

	//����5��sfm�õ������������ά�������ڲο�֡l֡���������ϵ�µģ�Ҳ������������ϵ�µģ���Ϊ��������û�����λ��
	//1: trangulate between l ----- frame_num - 1 ��һ�����ǵ�191�д��룬ѭ���ĵ�һ��
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			// ����һ֡��������λ�ˣ��������������ϵ��ο�֡����Ϊ��ʼֵ���е����Ż�����Ϊ˲ʱ�ĸı��С��
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			//solveFrameByPnP�������ֻ�������Ż����λ��R_initial��P_initial����l֡�ο�����ϵ�任����i֡����������õ����Ѿ������ǻ�������������ϵ�е�������֪���ĵ㱻��i֡���۲⵽
			//���õ��ⲿ��������֪�ĵ㣨����i=lʱ��һ������ó��ģ������if��������ǻ�)�����Ǳ���l֡������֡�����ӣ����Ҳһ���ᱻ����֮֡�������֡���۲⵽
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result 
		//�� ��l֡������֮֡���ĳһ֡i �� ����֡frame_num - 1����Ϊ��ʱ����֡����ڲο�ϵ��λ�˾���֪�� �������ǻ������i��frame_num - 1֡�й�ͬ�۲�����ά���꣨λ����������ϵ�µģ�������Ҳ�ǵ�l֡����ϵ��
		//�����ǻ�����Щ�㣬�ǲ�����l֡ �� i-1֮֡���֡���۲⵽�ģ�Ҳ��˵�Ǵ�i-1֡��i֡׷�ٽ����������Ӽ��ĵ��һ����
		//�������������е㱻���ǻ��ɹ�����������state=true
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2  ��l֡��֮���֡���ܻ��й�ͬ�۲��δ�����ǻ�
	//�ⲿ�ֵ��� ���ӵ�l֡����l+1֡׷�ٳɹ��ĵ㼯P��-���㼯P�б���l֡������֡�����ӵ��ӵ㼯P'����������֡�ڼ��ж�׷�ٵ��ӵ㼯
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp ������PnP�������Ǻ͵�l֡��PnP�������������׳��ֹ�ͬ�۲�㲻�����������������l֡Զ����Щ֡(ȷʵ�п��ܣ������������if�ж���䣩
		//����PnP�����õ���֪��ȵĵ㣨�豻i��l֡���ӣ����� ��i+1֡��l-1ÿһ֡ �� ��l֡ �����ӵĵ��е�һ���֣�����i֮֡ǰ��֡��һֱ���󱻳ɹ�׷�ٵ���l֡�ĵ㣬Ҳ������i֡�м���������һ���֣�Ҳ��һֱ׷�ٳɹ���
		//��ˣ���֡�͵�l֮֡���λ���������ɹ�����Ϊ���ӵ�������ܿ��ܲ�������Ϊʲô��ת��������֡��l֮֡ǰĳһ֡�����λ���أ���
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate  ����������ǻ������бȽϱ�Ҫ������iִ����ǰl-2֡��֡�����l֡�����ӵĵ㣬һ��Ҳ�ܱ���l-1֡�͵�l֡�����ӣ���������ֻ��Ҫ�Ե�l-1֡��l֡��һ�����ǻ��Ϳ����ˣ�
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points   
	//sfm_f�ﱣ�����������һЩ�����ᱻ��l֡���frame_num-1֡�۲�õ����ⲿ����Ҫ����������ǻ�����ʱ���йؼ�֡��λ�ˣ�����ڵ�l֡������֪
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			// ѡȡ�ܹ۲⵽�����������Զ����֡�������ǻ�����Ϊ��֮֡����Ҫ���ʵ���λ��
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			//��֪��֡��λ�˺�һ�Թ۲�ƥ��㣬���е�����ǻ�
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//���Ǵ���������ϵ���ο�֡l��ת������ǰ���֡
		//double array for ceres  ΪʲôҪ��ô�鷳����ת��Ԫ����λ�Ƹ��Ƶ�double�������У�Ӧ����ceres�в�����Ҫ��
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		//local_parameterization��������Ǹ��������c_rotation[i]�����ά�������������Ԫ����������Ӧ��������Ԫ��������
		//��AddParameterBlock()�����Ķ����У���һ������������Ӧ�������ã������Ϳ�����solveʱ�޸�c_rotation��c_translation��ֵ
		//·���������full BA���治�ٽ����Ż�
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)
		{
			//�趨��l֡�����������ϵ����תΪ�̶�ֵ��Ϊ0��������һ�����趨λ��Ϊ�̶�ֵ��ҲΪ0��
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{  //Ϊɶ��frame_num֡��Ե�l֡��λ��Ҫ�̶�����Ϊ0���������ת���̶���
		   //��Ϊ���ڵ�Ŀ�Ӿ����ԣ�Ҫ����ȫȷ��ϵͳ�Ľ⣬��Ҫ�̶�7�����ɶȣ�6λ�˺�һ���߶ȣ����г߶�������֡���λ�����̶�������Ҳ��ϵͳ��7�����ɹ�ά��
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		//ֻѡ����Щ�Ѿ����ǻ��ĵ����Full BA��Ϊɶ��ʱ����δ�����ǻ��ģ�
		//��Щ��ֻ�ڴ���֡�г��֣��۲���ʼ֡��������������û�б���׷�ٵ�����Щ�����slidewindow()�����б�ȥ����
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			//����һ֡���۲⵽��
			int l = sfm_f[i].observation[j].first;
			//��ʧ����
			//�۲���֡(j=0)Ҳ������ͶӰ����Ϊ�������ǵõ����ǵ�����������ϵ�µ����꣬����ͶӰ�����й۲�֡��
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
			// �˴�����в�飬ceres���Զ��Ѻ�������������4�� 3�� 3��������ǰ�����Ǵ����Ʊ���������ReprojectionError3D��()���غ����У�������ʽһ�£��������أ����Ա���òв
			//��������Ʋ����Ĵ��뷽ʽ��()���غ����д����Ʋ����Ĵ��뷽ʽҪһ��
			//��solveʱһ��AddParameterBlock����Ĵ����Ʊ����и��£���ô������update�в�ʱ����Ĳ����еĴ����Ʊ���Ҳ���Ѿ����µ��ˡ�ע�⣬sfm_f[i].position�����Ǵ���������������Ϊ�������Ѿ����Ƶ��㹻����
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	//
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;  if����Ŀ������ʲô����д��
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		//��Ϊ��i֡����������ϵ��Ҳ�ǵ�l֡����ϵ����λ������������ϵ����l֡����ϵ���µı�ʾ
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			//�����ڱ���sfm�гɹ����ǻ���������
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

