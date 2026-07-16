/**
 * @file avm.cpp
 * @brief 环视全景 (AVM) 系统主控模块
 */

#include "common.h"
#include "undistort.h"
#include "corner_detector.h"
#include "stitcher.h"
#include <iostream>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h> // 用于 Windows 上的 _mkdir
#endif

using namespace std;
using namespace cv;

// ========================================
// 全局变量定义与初始化
// ========================================

// 全局默认相机配置定义
const CameraConfig g_default_camera_config(
    0.43f,                                                      // fish_scale
    910.0f,                                                     // focal_length
    3.0f,                                                       // dx
    3.0f,                                                       // dy
    1280,                                                       // fish_width
    960,                                                        // fish_height
    1.55f,                                                      // undis_scale
    cv::Vec4d(-0.05611147, -0.05377447, 0.0115717, 0.0030788),  // fish2undis_params
    cv::Vec4d(0.18238692, -0.08579553, 0.03366532, -0.00561911) // undis2fish_params
);

// 四个方向的角点坐标
std::vector<cv::Point2f> g_corner_front; // 前视图角点
std::vector<cv::Point2f> g_corner_back;  // 后视图角点
std::vector<cv::Point2f> g_corner_left;  // 左视图角点
std::vector<cv::Point2f> g_corner_right; // 右视图角点

// 鸟瞰图中四个方向的目标角点坐标
std::vector<cv::Point2i> g_corner_bird_front; // 前视图鸟瞰角点
std::vector<cv::Point2i> g_corner_bird_back;  // 后视图鸟瞰角点
std::vector<cv::Point2i> g_corner_bird_left;  // 左视图鸟瞰角点
std::vector<cv::Point2i> g_corner_bird_right; // 右视图鸟瞰角点

// 四个方向的单应性矩阵
cv::Mat g_Homo_F; // 前视图单应性矩阵
cv::Mat g_Homo_B; // 后视图单应性矩阵
cv::Mat g_Homo_L; // 左视图单应性矩阵
cv::Mat g_Homo_R; // 右视图单应性矩阵

/**
 * @brief 环视系统主控程序入口
 * @description 完成完整的 AVM 处理流水线：图像去畸变、角点检测、透视变换、图像拼接
 * @return 程序执行状态
 */
int main()
{
    cout << "===========================================" << endl;
    cout << "[系统] 环视全景 (AVM) 系统已启动" << endl;
    cout << "===========================================" << endl;

    // 创建输出目录
    struct stat st = {0};
    if (stat(OUTPUT_DIR.c_str(), &st) == -1)
    {
#ifdef _WIN32
        if (_mkdir(OUTPUT_DIR.c_str()) == 0)
        {
#else
        if (mkdir(OUTPUT_DIR.c_str(), 0755) == 0)
        {
#endif
            cout << "[初始化] 已创建输出目录：" << OUTPUT_DIR << "/" << endl;
        }
        else
        {
            cout << "[警告] 无法创建输出目录，将使用当前目录" << endl;
        }
    }

    // 初始化角点容器
    g_corner_front = std::vector<cv::Point2f>(8);
    g_corner_back  = std::vector<cv::Point2f>(8);
    g_corner_left  = std::vector<cv::Point2f>(8);
    g_corner_right = std::vector<cv::Point2f>(8);

    cout << "[初始化] 相机参数初始化完成" << endl;

    // 读取四个方向的鱼眼图像
    cout << "[步骤 1] 正在读取鱼眼图像..." << endl;
    cv::Mat image_f = imread("assets/images/front.png");
    cv::Mat image_b = imread("assets/images/back.png");
    cv::Mat image_l = imread("assets/images/left.png");
    cv::Mat image_r = imread("assets/images/right.png");

    if (image_f.empty() || image_b.empty() || image_l.empty() || image_r.empty())
    {
        cout << "[错误] 无法读取输入图像文件" << endl;
        return -1;
    }
    cout << "[成功] 成功读取 4 张鱼眼图像" << endl;

    // 创建去畸变处理对象
    cout << "[步骤 2] 正在进行图像去畸变处理..." << endl;
    Undistort            undistort_handle(g_default_camera_config);
    std::vector<cv::Mat> undis2dis_front, undis2dis_back, undis2dis_left, undis2dis_right;

    // 执行图像去畸变
    cv::Mat front_undis = undistort_handle.undistort_func(image_f, undis2dis_front);
    cv::Mat back_undis  = undistort_handle.undistort_func(image_b, undis2dis_back);
    cv::Mat left_undis  = undistort_handle.undistort_func(image_l, undis2dis_left);
    cv::Mat right_undis = undistort_handle.undistort_func(image_r, undis2dis_right);

    // 复制一份干净的去畸变图像（没有绘制角点绿色圆圈）用于展示与鸟瞰图透视变换
    cv::Mat clean_front = front_undis.clone();
    cv::Mat clean_back  = back_undis.clone();
    cv::Mat clean_left  = left_undis.clone();
    cv::Mat clean_right = right_undis.clone();

    // 拼接四张去畸变图像方位并利用窗口显示出来
    showUndistortStitched(clean_front, clean_back, clean_left, clean_right);
    cout << "[成功] 去畸变图像方位拼接显示完成" << endl;

    // 检测标定板角点
    cout << "[步骤 3] 正在检测标定板角点..." << endl;
    bool success = true;
    success &= detectPoints(image_f, 20000, g_default_camera_config, g_corner_front, 0, ImageType::IMAGE_FRONT);
    success &= detectPoints(image_b, 20000, g_default_camera_config, g_corner_back, 0, ImageType::IMAGE_BACK);
    success &= detectPoints(image_l, 20000, g_default_camera_config, g_corner_left, 0, ImageType::IMAGE_LEFT);
    success &= detectPoints(image_r, 20000, g_default_camera_config, g_corner_right, 0, ImageType::IMAGE_RIGHT);

    if (!success)
    {
        cout << "[错误] 部分方向的标定板角点检测失败！" << endl;
        cout << "[提示] 请检查 `fish_scale` 缩放因子设置是否过小（当前为 " << g_default_camera_config.fish_scale
             << "f，导致标定板图案过小而无法识别）。建议将 `fish_scale` 设置为 0.42f ~ 0.50f 之间。" << endl;
        return -1;
    }
    else
    {
        cout << "[成功] 所有方向的角点检测已完成" << endl;
    }

    // 在去畸变图像上标记检测到的角点
    for (int j = 0; j < 8; j++)
    {
        circle(front_undis, g_corner_front[j], 3, cv::Scalar(0, 255, 0), 1);
        circle(back_undis, g_corner_back[j], 3, cv::Scalar(0, 255, 0), 1);
        circle(left_undis, g_corner_left[j], 3, cv::Scalar(0, 255, 0), 1);
        circle(right_undis, g_corner_right[j], 3, cv::Scalar(0, 255, 0), 1);
    }

    // 保存标记了角点的图像
    cv::imwrite(OUTPUT_DIR + "/front_undis_1.jpg", front_undis);
    cv::imwrite(OUTPUT_DIR + "/back_undis_1.jpg", back_undis);
    cv::imwrite(OUTPUT_DIR + "/left_undis_1.jpg", left_undis);
    cv::imwrite(OUTPUT_DIR + "/right_undis_1.jpg", right_undis);
    cout << "[成功] 角点标记图像已保存" << endl;

    // 初始化鸟瞰图参数
    init_params();

    // 计算单应性矩阵
    cout << "[步骤 4] 正在计算透视变换矩阵..." << endl;
    g_Homo_F = cv::findHomography(g_corner_front, g_corner_bird_front, 0);
    g_Homo_B = cv::findHomography(g_corner_back, g_corner_bird_back, 0);
    g_Homo_L = cv::findHomography(g_corner_left, g_corner_bird_left, 0);
    g_Homo_R = cv::findHomography(g_corner_right, g_corner_bird_right, 0);
    cout << "[成功] 透视变换矩阵计算完成" << endl;

    // 执行透视变换生成鸟瞰图
    cout << "[步骤 5] 正在生成鸟瞰图..." << endl;
    cv::Mat bird_front_image, bird_back_image, bird_left_image, bird_right_image;

    cv::warpPerspective(clean_front, bird_front_image, g_Homo_F, cv::Size(792, 305), cv::INTER_CUBIC);
    cv::warpPerspective(clean_back, bird_back_image, g_Homo_B, cv::Size(792, 305), cv::INTER_CUBIC);
    cv::warpPerspective(clean_left, bird_left_image, g_Homo_L, cv::Size(1131, 281), cv::INTER_CUBIC);
    cv::warpPerspective(clean_right, bird_right_image, g_Homo_R, cv::Size(1131, 281), cv::INTER_CUBIC);

    // 保存初始鸟瞰图图像
    cv::imwrite(OUTPUT_DIR + "/bird_front.jpg", bird_front_image);
    cv::imwrite(OUTPUT_DIR + "/bird_back.jpg", bird_back_image);
    cv::imwrite(OUTPUT_DIR + "/bird_left.jpg", bird_left_image);
    cv::imwrite(OUTPUT_DIR + "/bird_right.jpg", bird_right_image);
    cout << "[成功] 鸟瞰图生成已完成" << endl;

    // 旋转鸟瞰图以纠正朝向
    cout << "[步骤 6] 正在调整鸟瞰图朝向..." << endl;
    rotate(OUTPUT_DIR + "/bird_front.jpg", OUTPUT_DIR + "/bird_front_2.jpg", 0);
    rotate(OUTPUT_DIR + "/bird_back.jpg", OUTPUT_DIR + "/bird_back_2.jpg", 180);
    rotate(OUTPUT_DIR + "/bird_left.jpg", OUTPUT_DIR + "/bird_left_2.jpg", 90);
    rotate(OUTPUT_DIR + "/bird_right.jpg", OUTPUT_DIR + "/bird_right_2.jpg", -90);
    cout << "[成功] 鸟瞰图朝向调整已完成" << endl;

    // 执行全景拼接
    cout << "[步骤 7] 正在执行全景图像拼接..." << endl;
    join();

    cout << "===========================================" << endl;
    cout << "[系统] 环视全景系统处理完成" << endl;
    cout << "===========================================" << endl;
    return 0;
}