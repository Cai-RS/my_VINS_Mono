
//POSIX标准定义的unix类系统定义符号常量的头文件
#include <unistd.h>
//C语言标准库中用于输入输出的头文件
#include <stdio.h>
//stdlib.h是C标准函数库的头文件，声明了数值与字符串转换函数, 伪随机数生成函数, 动态内存分配函数, 进程控制函数等公共函数。
#include <stdlib.h>
//C标准函数库的头文件string.h头文件定义了一个变量类型、一个宏和各种操作字符数组的函数
#include <string.h>
#include <iostream>
#include <thread>
//iomanip 是一个用于操作 C++ 程序输出的库
#include <iomanip>
//cv.hpp和opencv.hpp是等同的关系。前者是早期opencv版本中的定义名称，而后者则是3.0版本之后的表示方法
#include <cv.h>
//opencv.hpp中己经包含了OpenCV各模块的头文件
#include <opencv2/opencv.hpp>
#include <highgui.h>
//稠密矩阵的代数运算(逆、特征值等)
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
//存储图片的路径
string sData_path = "/home/dataset/EuRoC/MH-05/mav0/";
//存储配置文件，包括imu和image的信息文件（imu文件中的是各个imu的时刻和数据；image文件中的是各个图像的时刻和名称（用于sData_path路径中找到图像））。
string sConfig_path = "../config/";

std::shared_ptr<System> pSystem;

void PubImuData()
{
	string sImu_data_file = sConfig_path + "MH_05_imu0.txt";
	cout << "1 PubImuData start sImu_data_file: " << sImu_data_file << endl;
	ifstream fsImu;
	//标准库的string类提供了3个成员函数来从一个string得到c类型的字符数组：c_str()、data()、copy(p,n)
	//c_str()的原型是：const char* c_str() const{}   
	//open函数的原型： void open( const char* filename, ios_base::openmode mode = ios_base::in ){}
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open())
	{
		cerr << "Failed to open imu file! " << sImu_data_file << endl;
		return;
	}

	std::string sImu_line;
	double dStampNSec = 0.0;
	Vector3d vAcc;
	Vector3d vGyr;
	//用C++标准库函数getline()从fsImu文件流中逐行读入赋值给sImu_line
	//当cin读取数据时，它会传递并忽略任何前导白色空格字符（空格、制表符或换行符）,
	//一旦它接触到第一个非空格字符即开始阅读，当它读取到下一个空白字符时，它将停止读取。因此若输入变量中间有空格符，就会出现读入错乱问题
	//为了解决这个问题，可以使用getline函数，它可读取整行，包括前导和嵌入的空格，并将其存储在字符串对象中。
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
	{
		//istringstream类的构造函数原形：istringstream::istringstream(string str){}；该类对象支持>>操作
		//为了使用istringstream类对象中的数据，需要使用str()函数将其变为字符串，或者用>>将其逐个放入某类型变量（包括字符串，会自动转换类型）
		// https://www.cnblogs.com/lsgxeva/p/8087148.html
		std::istringstream ssImuData(sImu_line);
		ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
		// cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
		pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
		//上面用System类中的PubImuData函数往Imu_buf中加入一个imu信息，然后对m_buf解锁
		//这里睡眠10ms，以便让其他线程获得m_buf的加锁。最开始应该是得让相机数据存入线程得到锁，然后才能让后端优化线程取得到数据
		usleep(5000*nDelayTimes); //sleep 10ms
	}
	fsImu.close();
}

void PubImageData()
{
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;
	
	// cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		pSystem->PubImageData(dStampNSec / 1e9, img);
		// cv::imshow("SOURCE IMAGE", img);
		// cv::waitKey(0);
		//往feature_buf中存入一帧图像数据之后，sleep 100ms（延时时间是imu的10倍，因为两帧图像之间有许多imu数据）
		usleep(50000*nDelayTimes);
	}
	fsImage.close();
}

#ifdef __APPLE__
// support for MacOS
void DrawIMGandGLinMainThrd(){
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;

	pSystem->InitDrawGL();
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		//pSystem->PubImageData(dStampNSec / 1e9, img);
		cv::Mat show_img;
		cv::cvtColor(img, show_img, CV_GRAY2RGB);
		if (SHOW_TRACK)
		{
			for (unsigned int j = 0; j < pSystem->trackerData[0].cur_pts.size(); j++)
			{
				double len = min(1.0, 1.0 *  pSystem->trackerData[0].track_cnt[j] / WINDOW_SIZE);
				cv::circle(show_img,  pSystem->trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
			}

			cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
			cv::imshow("IMAGE", show_img);
		  // cv::waitKey(1);
		}

		pSystem->DrawGLFrame();
		usleep(50000*nDelayTimes);
	}
	fsImage.close();

} 
#endif

// main函数是一个工程中的初始入口（第一个被执行的函数）
int main(int argc, char **argv)
{
	if(argc != 3)
	{
		cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n" 
			<< "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
		return -1;
	}
	sData_path = argv[1];
	sConfig_path = argv[2];

	//p.reset(q) 会令智能指针p中存放指针q，即p指向q的空间，而且会释放原来的空间（默认是delete）
	pSystem.reset(new System(sConfig_path));
	
	//下面三个线程会竞争m_buf这个锁，thd_BackEnd中需要从队列中取数据，另外两个要向队列中放入数据
	// 另外，thd_BackEnd线程中还有m_estimator这个锁，用于后端的估计（m_estimator不被另外两个线程共用，说明后端优化部分需要用到的资源与另两个线程无关）
	//pSystem是作为参数？？
	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);
		
	// sleep(5);
	std::thread thd_PubImuData(PubImuData);

	std::thread thd_PubImageData(PubImageData);

#ifdef __linux__	
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
	DrawIMGandGLinMainThrd();
#endif
	//为什么这里不需要对thd_BackEnd线程进行阻塞？因为在退出main函数前，pSystem的析构函数会被调用
	//而在~System()函数中有设定 bStart_backend=false （此为后端线程函数的执行判断条件），因此后端优化线程在主线程结束后也不会开始下一次优化（虽然会再调用一次后端优化函数）;
	//另外，~System()函数中还有m_estimator.lock()的操作，因此最线程会等待最近的一次后端优化线程执行完毕后，才能完成pSystem对象的析构。
	//这也就说明了，在两个数据存入线程都结束之后，和主线程结束之前，必须保证后端优化至少再完整执行一次（所以每次存图像之后，会sleep 100ms，这个时间应该足够后端线程再完整执行一次？）
	thd_PubImuData.join();
	thd_PubImageData.join();

	// thd_BackEnd.join();
	// thd_Draw.join();

	cout << "main end... see you ..." << endl;
	return 0;
}
