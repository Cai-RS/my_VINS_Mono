#pragma once
// #include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

//����û�д�extern�ı�������ͬʱҲ���Ƕ��壨ָ����洢�ռ䣩�� �� ��int a;�����Ǽ������ֶ���
//externʹ�ñ����������Ͷ����ǿ��Էֿ��ģ�����ֻ�����˱���������
//������ڱ���ļ��ж����ʹ�������ȫ�ֱ�������Ҫ���������ǣ���include���ͷ�ļ���
extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;

//NUM_OF_CAM��������Ѿ������ˡ�const�����趨Ϊ�����ļ�����Ч��������ļ��г���ͬ����const����ʱ����ʵ��ͬ���ڲ�ͬ�ļ��зֱ����˶����ı����������ظ����壡
const int NUM_OF_CAM = 1;

extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

void readParameters();
