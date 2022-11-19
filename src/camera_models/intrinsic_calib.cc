//boost是完全免费的 C++ 程序库，共包含 160 余个库/组件，涵盖字符串与文本处理、容器、迭代器、算法、图像处理、模板元编程、并发编程等多个领域，使用 Boost，将大大增强 C++ 的功能和表现力。
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
// iomanip是I/O流控制头文件,就像C里面的格式化输出一样,例如设置输出的浮点数的有效位数（默认为6位）
#include <iomanip>
#include <iostream>
//algorithm是所有STL头文件中最大的一个，它是由一大堆模版函数组成的，常用到的功能范围涉及到比较、交换、查找、遍历操作、复制等等
#include <algorithm>
//opencv中的core核心模块详解——定义了基础数据类型以及算法，如Point,Size(宽和高）,Rect,Scalar(即四维点）,Vec,Matx,Range类等
#include <opencv2/core/core.hpp>
//improc模块包含图像处理和视频处理算法
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
    // 程序参数项(program options)是一系列pair<name,value>，program_options允许程序开发者获得通过命令行(command line)和配置文件(config file)获取这些参数项。
    // 构造选项描述器,其参数为该描述器的名字
    boost::program_options::options_description desc("Allowed options");
    // 为选项描述器增加选项，其参数依次为: key, value的类型（输入时要遵守的格式）和默认值，该选项的描述
    desc.add_options()
        ("help", "produce help message")
        //width是长项目名称，w是短名称，用户均可使用
        ("width,w", boost::program_options::value<int>(&boardSize.width)->default_value(8), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", boost::program_options::value<int>(&boardSize.height)->default_value(12), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", boost::program_options::value<float>(&squareSize)->default_value(7.f), "Size of one square in mm")
        //下面这个选项是输入数据的路径
        ("input,i", boost::program_options::value<std::string>(&inputDir)->default_value("calibrationdata"), "Input directory containing chessboard images")
        //图像前缀（prefix）
        ("prefix,p", boost::program_options::value<std::string>(&prefix)->default_value("left-"), "Prefix of images")
        ("file-extension,e", boost::program_options::value<std::string>(&fileExtension)->default_value(".png"), "File extension of images")
        ("camera-model", boost::program_options::value<std::string>(&cameraModel)->default_value("mei"), "Camera model: kannala-brandt | mei | pinhole")
        ("camera-name", boost::program_options::value<std::string>(&cameraName)->default_value("camera"), "Name of camera")
        ("opencv", boost::program_options::bool_switch(&useOpenCV)->default_value(true), "Use OpenCV to detect corners")
        ("view-results", boost::program_options::bool_switch(&viewResults)->default_value(false), "View results")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(true), "Verbose output")
        ;
    //创建位置参数项对象。位置参数项目的是告诉系统当输入的参数指前面缺少项目名称时，直接认定为规定中的哪一项
    boost::program_options::positional_options_description pdesc;
    // 指出所有的“位置参数项”应被翻译成“input”项
    pdesc.add("input", 1);

    //创建选项存储器,继承自map容器
    boost::program_options::variables_map vm;
    //存储命令行
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    //解析命令行
    boost::program_options::notify(vm);

    //如果在命令行或配置文件中给出 选项"help"这个名字，则执行以下命令。也可以输入其他的选项的name和value赋值给map容器对象vm，然后再执行相关命令
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    //找不到输入路径
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
    //directory_iterator类：获取文件系统目录中文件的迭代器容器，其元素为directory_entry对象可用于遍历目录
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        //如果不满足条件，则忽略后面的语句跳过此次for循环，直接开始下次循环
        //file_status类：用于获取和修改文件（或目录）的属性
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }
        //filename为完整的文件名，包括 前缀+‘.’+后缀
        std::string filename = itr->path().filename().string();

        // check if prefix matches 检查文件名前缀是否符合给定输入选项prefix
        if (!prefix.empty())
        {
            //int compare (size_type pos, size_type n, const basic_string& s) const; 将子串（从第0位开始，长度为n）和定量字符串s进行比较
            if (filename.compare(0, prefix.length(), prefix) != 0)
            {
                continue;
            }
        }

        // check if file extension matches 检查文件扩展名（后缀）是否符合给定输入选项fileExtension
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
    // sort函数用于对给定区间所有元素进行排序，默认为升序
    std::sort(imageFilenames.begin(), imageFilenames.end());
    //读入第一幅图像
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
            //在图像中某一位置显示文字，cbImageFilenames.at(i)即为被显示的文字，cv::Point(10,20)为文字的位置
            cv::putText(cbImages.at(i), cbImageFilenames.at(i), cv::Point(10,20),
                        cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                        1, CV_AA);
            cv::imshow("Image", cbImages.at(i));
            cv::waitKey(0);
        }
    }

    return 0;
}
