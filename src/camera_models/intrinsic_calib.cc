//boost����ȫ��ѵ� C++ ����⣬������ 160 �����/����������ַ������ı��������������������㷨��ͼ����ģ��Ԫ��̡�������̵ȶ������ʹ�� Boost���������ǿ C++ �Ĺ��ܺͱ�������
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
// iomanip��I/O������ͷ�ļ�,����C����ĸ�ʽ�����һ��,������������ĸ���������Чλ����Ĭ��Ϊ6λ��
#include <iomanip>
#include <iostream>
//algorithm������STLͷ�ļ�������һ����������һ���ģ�溯����ɵģ����õ��Ĺ��ܷ�Χ�漰���Ƚϡ����������ҡ��������������Ƶȵ�
#include <algorithm>
//opencv�е�core����ģ����⡪�������˻������������Լ��㷨����Point,Size(��͸ߣ�,Rect,Scalar(����ά�㣩,Vec,Matx,Range���
#include <opencv2/core/core.hpp>
//improcģ�����ͼ�������Ƶ�����㷨
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camodocal/chessboard/Chessboard.h"
#include "camodocal/calib/CameraCalibration.h"
#include "camodocal/gpl/gpl.h"

int main(int argc, char** argv)
{
    cv::Size boardSize;
    float squareSize;
    std::string inputDir;
    std::string cameraModel;
    std::string cameraName;
    std::string prefix;
    std::string fileExtension;
    bool useOpenCV;
    bool viewResults;
    bool verbose;

    //========= Handling Program options =========
    // ���������(program options)��һϵ��pair<name,value>��program_options������򿪷��߻��ͨ��������(command line)�������ļ�(config file)��ȡ��Щ�����
    // ����ѡ��������,�����Ϊ��������������
    boost::program_options::options_description desc("Allowed options");
    // Ϊѡ������������ѡ����������Ϊ: key, value�����ͣ�����ʱҪ���صĸ�ʽ����Ĭ��ֵ����ѡ�������
    desc.add_options()
        ("help", "produce help message")
        //width�ǳ���Ŀ���ƣ�w�Ƕ����ƣ��û�����ʹ��
        ("width,w", boost::program_options::value<int>(&boardSize.width)->default_value(8), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", boost::program_options::value<int>(&boardSize.height)->default_value(12), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", boost::program_options::value<float>(&squareSize)->default_value(7.f), "Size of one square in mm")
        //�������ѡ�����������ݵ�·��
        ("input,i", boost::program_options::value<std::string>(&inputDir)->default_value("calibrationdata"), "Input directory containing chessboard images")
        //ͼ��ǰ׺��prefix��
        ("prefix,p", boost::program_options::value<std::string>(&prefix)->default_value("left-"), "Prefix of images")
        ("file-extension,e", boost::program_options::value<std::string>(&fileExtension)->default_value(".png"), "File extension of images")
        ("camera-model", boost::program_options::value<std::string>(&cameraModel)->default_value("mei"), "Camera model: kannala-brandt | mei | pinhole")
        ("camera-name", boost::program_options::value<std::string>(&cameraName)->default_value("camera"), "Name of camera")
        ("opencv", boost::program_options::bool_switch(&useOpenCV)->default_value(true), "Use OpenCV to detect corners")
        ("view-results", boost::program_options::bool_switch(&viewResults)->default_value(false), "View results")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(true), "Verbose output")
        ;
    //����λ�ò��������λ�ò�����Ŀ���Ǹ���ϵͳ������Ĳ���ָǰ��ȱ����Ŀ����ʱ��ֱ���϶�Ϊ�涨�е���һ��
    boost::program_options::positional_options_description pdesc;
    // ָ�����еġ�λ�ò����Ӧ������ɡ�input����
    pdesc.add("input", 1);

    //����ѡ��洢��,�̳���map����
    boost::program_options::variables_map vm;
    //�洢������
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    //����������
    boost::program_options::notify(vm);

    //����������л������ļ��и��� ѡ��"help"������֣���ִ���������Ҳ��������������ѡ���name��value��ֵ��map��������vm��Ȼ����ִ���������
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    //�Ҳ�������·��
    if (!boost::filesystem::exists(inputDir) && !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: Cannot find input directory " << inputDir << "." << std::endl;
        return 1;
    }

    camodocal::Camera::ModelType modelType;
    if (boost::iequals(cameraModel, "kannala-brandt"))
    {
        modelType = camodocal::Camera::KANNALA_BRANDT;
    }
    else if (boost::iequals(cameraModel, "mei"))
    {
        modelType = camodocal::Camera::MEI;
    }
    else if (boost::iequals(cameraModel, "pinhole"))
    {
        modelType = camodocal::Camera::PINHOLE;
    }
    else if (boost::iequals(cameraModel, "scaramuzza"))
    {
        modelType = camodocal::Camera::SCARAMUZZA;
    }
    else
    {
        std::cerr << "# ERROR: Unknown camera model: " << cameraModel << std::endl;
        return 1;
    }

    switch (modelType)
    {
    case camodocal::Camera::KANNALA_BRANDT:
        std::cout << "# INFO: Camera model: Kannala-Brandt" << std::endl;
        break;
    case camodocal::Camera::MEI:
        std::cout << "# INFO: Camera model: Mei" << std::endl;
        break;
    case camodocal::Camera::PINHOLE:
        std::cout << "# INFO: Camera model: Pinhole" << std::endl;
        break;
    case camodocal::Camera::SCARAMUZZA:
        std::cout << "# INFO: Camera model: Scaramuzza-Omnidirect" << std::endl;
        break;
    }

    // look for images in input directory
    std::vector<std::string> imageFilenames;
    //directory_iterator�ࣺ��ȡ�ļ�ϵͳĿ¼���ļ��ĵ�������������Ԫ��Ϊdirectory_entry��������ڱ���Ŀ¼
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        //�������������������Ժ������������˴�forѭ����ֱ�ӿ�ʼ�´�ѭ��
        //file_status�ࣺ���ڻ�ȡ���޸��ļ�����Ŀ¼��������
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }
        //filenameΪ�������ļ��������� ǰ׺+��.��+��׺
        std::string filename = itr->path().filename().string();

        // check if prefix matches ����ļ���ǰ׺�Ƿ���ϸ�������ѡ��prefix
        if (!prefix.empty())
        {
            //int compare (size_type pos, size_type n, const basic_string& s) const; ���Ӵ����ӵ�0λ��ʼ������Ϊn���Ͷ����ַ���s���бȽ�
            if (filename.compare(0, prefix.length(), prefix) != 0)
            {
                continue;
            }
        }

        // check if file extension matches ����ļ���չ������׺���Ƿ���ϸ�������ѡ��fileExtension
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
        {
            continue;
        }

        imageFilenames.push_back(itr->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenames.back() << std::endl;
        }
    }

    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: # images: " << imageFilenames.size() << std::endl;
    }
    // sort�������ڶԸ�����������Ԫ�ؽ�������Ĭ��Ϊ����
    std::sort(imageFilenames.begin(), imageFilenames.end());
    //�����һ��ͼ��
    cv::Mat image = cv::imread(imageFilenames.front(), -1);
    const cv::Size frameSize = image.size();

    camodocal::CameraCalibration calibration(modelType, cameraName, frameSize, boardSize, squareSize);
    calibration.setVerbose(verbose);

    std::vector<bool> chessboardFound(imageFilenames.size(), false);
    for (size_t i = 0; i < imageFilenames.size(); ++i)
    {
        image = cv::imread(imageFilenames.at(i), -1);

        camodocal::Chessboard chessboard(boardSize, image);

        chessboard.findCorners(useOpenCV);
        if (chessboard.cornersFound())
        {
            if (verbose)
            {
                std::cerr << "# INFO: Detected chessboard in image " << i + 1 << ", " << imageFilenames.at(i) << std::endl;
            }

            calibration.addChessboardData(chessboard.getCorners());

            cv::Mat sketch;
            chessboard.getSketch().copyTo(sketch);

            cv::imshow("Image", sketch);
            cv::waitKey(50);
        }
        else if (verbose)
        {
            std::cerr << "# INFO: Did not detect chessboard in image " << i + 1 << std::endl;
        }
        chessboardFound.at(i) = chessboard.cornersFound();
    }
    cv::destroyWindow("Image");

    if (calibration.sampleCount() < 10)
    {
        std::cerr << "# ERROR: Insufficient number of detected chessboards." << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: Calibrating..." << std::endl;
    }

    double startTime = camodocal::timeInSeconds();

    calibration.calibrate();
    calibration.writeParams(cameraName + "_camera_calib.yaml");
    calibration.writeChessboardData(cameraName + "_chessboard_data.dat");

    if (verbose)
    {
        std::cout << "# INFO: Calibration took a total time of "
                  << std::fixed << std::setprecision(3) << camodocal::timeInSeconds() - startTime
                  << " sec.\n";
    }

    if (verbose)
    {
        std::cerr << "# INFO: Wrote calibration file to " << cameraName + "_camera_calib.yaml" << std::endl;
    }

    if (viewResults)
    {
        std::vector<cv::Mat> cbImages;
        std::vector<std::string> cbImageFilenames;

        for (size_t i = 0; i < imageFilenames.size(); ++i)
        {
            if (!chessboardFound.at(i))
            {
                continue;
            }

            cbImages.push_back(cv::imread(imageFilenames.at(i), -1));
            cbImageFilenames.push_back(imageFilenames.at(i));
        }

        // visualize observed and reprojected points
        calibration.drawResults(cbImages);

        for (size_t i = 0; i < cbImages.size(); ++i)
        {
            //��ͼ����ĳһλ����ʾ���֣�cbImageFilenames.at(i)��Ϊ����ʾ�����֣�cv::Point(10,20)Ϊ���ֵ�λ��
            cv::putText(cbImages.at(i), cbImageFilenames.at(i), cv::Point(10,20),
                        cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                        1, CV_AA);
            cv::imshow("Image", cbImages.at(i));
            cv::waitKey(0);
        }
    }

    return 0;
}
