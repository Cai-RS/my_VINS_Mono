#include "initial/initial_sfm.h"

GlobalSFM::GlobalSFM(){}
//上面为构造函数的定义

//此函数为已知某特征点在两帧的观测，采用求逆的方法来求解点的三维（世界）坐标，因为只有两次观测，刚好满秩（非超定，不需要最小二乘）
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	//同一个点在不同帧Pose0和Pose1（指的是世界坐标系或者参考帧坐标系到该帧的转换矩阵）下的观测值point0和point1，需要用三角化求解该点在世界坐标系或者第一帧中的坐标（归一化，4维）
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	//求解矩阵的SVD分解，分解得到的右奇异矩阵的最右列就是待求的向量，即奇异值最小对应的右奇异向量。下面是将向量的第四个量化为1
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}



//参数里的i应该指的是第i帧图像，此函数为已知特征点在第i帧中的观测和它在世界（或参考帧）坐标系下的三维坐标，给定优化初始值，用BA来优化相对位姿（这里并未优化三维坐标）
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	//j为特征的个数（不重复）
	for (int j = 0; j < feature_num; j++)
	{
		//需要找到深度（表达在参考坐标系下）已经被估计出来的点
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		//sfm_f中的每个元素代表了一个特征点出现的帧及其对应二维观察，即j代表某个特征点，k代表该点出现在哪些帧，这里遍历j和k时为了把第i帧中观察到的所有特侦点信息提取出来
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				//某个id的点在第i帧最多只会被观测到一次，因此如果找到了观测，直接退出循环
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
	//K应该是相机的内参矩阵？D应该是用来储存点的深度值？revc是初值旋转矩阵转化为cv的格式后再变化为旋转向量的格式，t同理。要迭代优化D, rvec, t
	bool pnp_succ;
	//solvePnP已知条件为控制点 A,B,C在世界（参考）坐标系下的位置 以及 在摄像机中的投影坐标。PnP问题至少需要三组点，即P3P
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//旋转向量转化为cv格式的旋转矩阵
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}



// 此函数为已知某两帧的相对（于世界或初始帧坐标系）位姿时，利用这两帧的位姿来进行两帧中共同观测点的三角化（计算它们在世界坐标系下的坐标），这里应该三角化尽可能多的点，以便这两帧之间其他帧的PnP能够顺利
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	// 判断为不同帧，对应第200行会出现两帧都是第frame_num的情况
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)
	{
		//为什么这里是true时反而不进行计算？true表示该路标点的三维坐标（在世界坐标系或者初始帧下）已被求解，对应第200行，有些在第i和第frame_num-1帧中的共同观测点已经被三角化了，只需三角化新的共同点
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
// 参数里的l为被选定初始化的参考帧id，它是通过计算与最新帧具有满足条件的视差和公共点的旧帧，relative_R和relative_T就是它和最新帧的相对位姿
//frame_num这里的值等于frame_count+1，而frame_count=WINDOW_SIZE。因此frame_num=WINDOW_SIZE+1。即frame_num表示的是数组的长度！
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view  先对参考帧l和最新帧进行初始化
	// 这里的q[l]为指定的帧，但是为什么它相对世界（或初始帧）坐标系没有旋转和平移呢？说明它就被设置为和世界坐标系的相对位姿为0，这点在后面的位姿优化时的固定位姿要对应
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	//q数组的长度为frame=WINDOW_SIZE+1，因此它的最大元素索引为frame_num - 1，即最新帧
	//relative_R将最新帧相机坐标系旋转到世界坐标系
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	//四元数数组
	Quaterniond c_Quat[frame_num];
	// c_rotation为四元数维度的数组，但不一定为四元数，需手动保证。用于ceres中，格式要求
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	// 从这里求逆来看，上面的输入q表示从某帧相机坐标系转到参考帧相机坐标系（第l帧）的四元数，因为下面要输入triangulateTwoFrames的位姿参数Pose就是从参考坐标系转到相机坐标系
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

	//以下5步sfm得到的特征点的三维坐标是在参考帧l帧相机作坐标系下的，也即是世界坐标系下的，因为这里两者没有相对位姿
	//1: trangulate between l ----- frame_num - 1 这一步就是第191行代码，循环的第一步
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			// 以上一帧相机的相对位姿（相对于世界坐标系或参考帧）作为初始值进行迭代优化（因为瞬时的改变很小）
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			//solveFrameByPnP这个函数只是用于优化求解位姿R_initial和P_initial（第l帧参考坐标系变换到第i帧相机），利用的是已经被三角化（在世界坐标系中的坐标已知）的点被第i帧所观测到
			//利用的这部分坐标已知的点（是在i=l时的一步计算得出的，即这个if后面的三角化)，它们被第l帧和最新帧所共视，因此也一定会被这两帧之间的其他帧所观测到
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result 
		//用 第l帧和最新帧之间的某一帧i 与 最新帧frame_num - 1（因为此时这俩帧相对于参考系的位姿均已知） 进行三角化，求得i与frame_num - 1帧中共同观测点的三维坐标（位于世界坐标系下的，假设上也是第l帧坐标系）
		//被三角化的这些点，是不被第l帧 至 i-1帧之间的帧所观测到的，也就说是从i-1帧到i帧追踪结束后新增加检测的点的一部分
		//在这个函数里，当有点被三角化成功，会设置其state=true
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2  第l帧和之后的帧可能还有共同观测点未被三角化
	//这部分点是 （从第l帧到第l+1帧追踪成功的点集P）-（点集P中被第l帧和最新帧所共视的子点集P'），即在两帧期间中断追踪的子点集
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp 这里先PnP，但都是和第l帧做PnP，这样不会容易出现共同观测点不足的情况吗？尤其是离第l帧远的那些帧(确实有可能，所以有下面的if判断语句）
		//这里PnP所利用的已知深度的点（需被i和l帧共视），是 第i+1帧至l-1每一帧 与 第l帧 都共视的点中的一部分，包括i帧之前的帧中一直往后被成功追踪到第l帧的点，也包括第i帧中检测新增点的一部分（也需一直追踪成功）
		//因此，首帧和第l帧之间的位姿最难求解成功，因为共视点的数量很可能不够（那为什么不转而先求首帧与l帧之前某一帧的相对位姿呢？）
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate  下面这个三角化操作有比较必要对所有i执行吗？前l-2帧各帧中与第l帧所共视的点，一定也能被第l-1帧和第l帧所共视，所以这里只需要对第l-1帧和l帧做一次三角化就可以了？
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points   
	//sfm_f里保存的特征点有一些并不会被第l帧或第frame_num-1帧观测得到，这部分需要另外进行三角化。这时所有关键帧的位姿（相对于第l帧）均已知
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			// 选取能观测到该特征点的最远的两帧进行三角化，因为两帧之间需要有适当的位移
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			//已知两帧的位姿和一对观测匹配点，进行点的三角化
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
		//均是从世界坐标系（参考帧l）转换到当前相机帧
		//double array for ceres  为什么要这么麻烦把旋转四元数和位移复制到double型数组中？应该是ceres中参数的要求？
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		//local_parameterization这个变量是告诉求解器c_rotation[i]这个四维参数代表的是四元数，其运算应该满足四元数的性质
		//在AddParameterBlock()函数的定义中，第一个参数的类型应该是引用，这样就可以在solve时修改c_rotation和c_translation的值
		//路标点坐标在full BA里面不再进行优化
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)
		{
			//设定第l帧相对世界坐标系的旋转为固定值（为0），下面一句是设定位移为固定值（也为0）
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{  //为啥第frame_num帧相对第l帧的位移要固定（不为0）而相对旋转不固定？
		   //因为对于单目视觉而言，要想完全确定系统的解，需要固定7个自由度（6位姿和一个尺度，其中尺度由最新帧相对位移来固定），这也是系统的7个不可观维度
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		//只选择那些已经三角化的点进行Full BA。为啥这时仍有未被三角化的？
		//有些点只在次新帧中出现（观测起始帧），但在最新中没有被被追踪到（这些点会在slidewindow()函数中被去除）
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			//在哪一帧被观测到的
			int l = sfm_f[i].observation[j].first;
			//损失函数
			//观测首帧(j=0)也具有重投影误差，因为上面我们得到的是点在世界坐标系下的坐标，可以投影到所有观测帧上
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
			// 此处加入残差块，ceres会自动把后面三个参数（4， 3， 3）（其中前两个是待估计变量）代入ReprojectionError3D的()重载函数中（参数格式一致，引起重载），以便求得残差。
			//这里待估计参数的传入方式和()重载函数中待估计参数的传入方式要一致
			//在solve时一旦AddParameterBlock传入的待估计变量有更新，那么在这里update残差时传入的参数中的待估计变量也是已经更新的了。注意，sfm_f[i].position并不是待估计量，这里认为点坐标已经估计得足够好了
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
		//cout << "vision only BA converge" << endl;  if后面的框里可以什么都不写。
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
		//此为第i帧到世界坐标系（也是第l帧坐标系）的位移在世界坐标系（第l帧坐标系）下的表示
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			//最终在本轮sfm中成功三角化的特征点
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

