
//POSIX��׼�����unix��ϵͳ������ų�����ͷ�ļ�
#include <unistd.h>
//C���Ա�׼�����������������ͷ�ļ�
#include <stdio.h>
//stdlib.h��C��׼�������ͷ�ļ�����������ֵ���ַ���ת������, α��������ɺ���, ��̬�ڴ���亯��, ���̿��ƺ����ȹ���������
#include <stdlib.h>
//C��׼�������ͷ�ļ�string.hͷ�ļ�������һ���������͡�һ����͸��ֲ����ַ�����ĺ���
#include <string.h>
#include <iostream>
#include <thread>
//iomanip ��һ�����ڲ��� C++ ��������Ŀ�
#include <iomanip>
//cv.hpp��opencv.hpp�ǵ�ͬ�Ĺ�ϵ��ǰ��������opencv�汾�еĶ������ƣ�����������3.0�汾֮��ı�ʾ����
#include <cv.h>
//opencv.hpp�м���������OpenCV��ģ���ͷ�ļ�
#include <opencv2/opencv.hpp>
#include <highgui.h>
//���ܾ���Ĵ�������(�桢����ֵ��)
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
//�洢ͼƬ��·��
string sData_path = "/home/dataset/EuRoC/MH-05/mav0/";
//�洢�����ļ�������imu��image����Ϣ�ļ���imu�ļ��е��Ǹ���imu��ʱ�̺����ݣ�image�ļ��е��Ǹ���ͼ���ʱ�̺����ƣ�����sData_path·�����ҵ�ͼ�񣩣���
string sConfig_path = "../config/";

std::shared_ptr<System> pSystem;

void PubImuData()
{
	string sImu_data_file = sConfig_path + "MH_05_imu0.txt";
	cout << "1 PubImuData start sImu_data_file: " << sImu_data_file << endl;
	ifstream fsImu;
	//��׼���string���ṩ��3����Ա��������һ��string�õ�c���͵��ַ����飺c_str()��data()��copy(p,n)
	//c_str()��ԭ���ǣ�const char* c_str() const{}   
	//open������ԭ�ͣ� void open( const char* filename, ios_base::openmode mode = ios_base::in ){}
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
	//��C++��׼�⺯��getline()��fsImu�ļ��������ж��븳ֵ��sImu_line
	//��cin��ȡ����ʱ�����ᴫ�ݲ������κ�ǰ����ɫ�ո��ַ����ո��Ʊ�����з���,
	//һ�����Ӵ�����һ���ǿո��ַ�����ʼ�Ķ���������ȡ����һ���հ��ַ�ʱ������ֹͣ��ȡ���������������м��пո�����ͻ���ֶ����������
	//Ϊ�˽��������⣬����ʹ��getline���������ɶ�ȡ���У�����ǰ����Ƕ��Ŀո񣬲�����洢���ַ��������С�
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
	{
		//istringstream��Ĺ��캯��ԭ�Σ�istringstream::istringstream(string str){}���������֧��>>����
		//Ϊ��ʹ��istringstream������е����ݣ���Ҫʹ��str()���������Ϊ�ַ�����������>>�����������ĳ���ͱ����������ַ��������Զ�ת�����ͣ�
		// https://www.cnblogs.com/lsgxeva/p/8087148.html
		std::istringstream ssImuData(sImu_line);
		ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
		// cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
		pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
		//������System���е�PubImuData������Imu_buf�м���һ��imu��Ϣ��Ȼ���m_buf����
		//����˯��10ms���Ա��������̻߳��m_buf�ļ������ʼӦ���ǵ���������ݴ����̵߳õ�����Ȼ������ú���Ż��߳�ȡ�õ�����
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
		//��feature_buf�д���һ֡ͼ������֮��sleep 100ms����ʱʱ����imu��10������Ϊ��֡ͼ��֮�������imu���ݣ�
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

// main������һ�������еĳ�ʼ��ڣ���һ����ִ�еĺ�����
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

	//p.reset(q) ��������ָ��p�д��ָ��q����pָ��q�Ŀռ䣬���һ��ͷ�ԭ���Ŀռ䣨Ĭ����delete��
	pSystem.reset(new System(sConfig_path));
	
	//���������̻߳Ὰ��m_buf�������thd_BackEnd����Ҫ�Ӷ�����ȡ���ݣ���������Ҫ������з�������
	// ���⣬thd_BackEnd�߳��л���m_estimator����������ں�˵Ĺ��ƣ�m_estimator�������������̹߳��ã�˵������Ż�������Ҫ�õ�����Դ���������߳��޹أ�
	//pSystem����Ϊ��������
	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);
		
	// sleep(5);
	std::thread thd_PubImuData(PubImuData);

	std::thread thd_PubImageData(PubImageData);

#ifdef __linux__	
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
	DrawIMGandGLinMainThrd();
#endif
	//Ϊʲô���ﲻ��Ҫ��thd_BackEnd�߳̽�����������Ϊ���˳�main����ǰ��pSystem�����������ᱻ����
	//����~System()���������趨 bStart_backend=false ����Ϊ����̺߳�����ִ���ж�����������˺���Ż��߳������߳̽�����Ҳ���Ὺʼ��һ���Ż�����Ȼ���ٵ���һ�κ���Ż�������;
	//���⣬~System()�����л���m_estimator.lock()�Ĳ�����������̻߳�ȴ������һ�κ���Ż��߳�ִ����Ϻ󣬲������pSystem�����������
	//��Ҳ��˵���ˣ����������ݴ����̶߳�����֮�󣬺����߳̽���֮ǰ�����뱣֤����Ż�����������ִ��һ�Σ�����ÿ�δ�ͼ��֮�󣬻�sleep 100ms�����ʱ��Ӧ���㹻����߳�������ִ��һ�Σ���
	thd_PubImuData.join();
	thd_PubImageData.join();

	// thd_BackEnd.join();
	// thd_Draw.join();

	cout << "main end... see you ..." << endl;
	return 0;
}
