/**
 * @file common.h
 * @brief 公共通用定义与全局变量声明
 */

#ifndef AVM_COMMON_H
#define AVM_COMMON_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 常量定义
#define IMAGE_BACK_PIXEL_Y  643 // 拼接结果中后视图图像的 Y 轴偏移量
#define IMAGE_RIGHT_PIXEL_X 398 // 拼接结果中右视图图像的 X 轴偏移量

/**
 * @brief 图像类型枚举
 */
typedef enum
{
    IMAGE_FRONT = 0, // 前视图图像
    IMAGE_BACK,      // 后视图图像
    IMAGE_LEFT,      // 左视图图像
    IMAGE_RIGHT      // 右视图图像
} ImageType;

// 全局变量声明（由 src/avm.cpp 定义）
extern cv::Mat   g_intrinsic_undis;   // 去畸变内参矩阵
extern cv::Mat   g_intrinsic;         // 原始内参矩阵
extern cv::Vec4d g_fish2undis_params; // 鱼眼到无畸变变换参数

// 四个方向的角点坐标声明
extern std::vector<cv::Point2f> g_corner_front; // 前视图角点
extern std::vector<cv::Point2f> g_corner_back;  // 后视图角点
extern std::vector<cv::Point2f> g_corner_left;  // 左视图角点
extern std::vector<cv::Point2f> g_corner_right; // 右视图角点

// 鸟瞰图中四个方向的目标角点坐标声明
extern std::vector<cv::Point2i> g_corner_bird_front; // 前视图鸟瞰角点
extern std::vector<cv::Point2i> g_corner_bird_back;  // 后视图鸟瞰角点
extern std::vector<cv::Point2i> g_corner_bird_left;  // 左视图鸟瞰角点
extern std::vector<cv::Point2i> g_corner_bird_right; // 右视图鸟瞰角点

// 四个方向 of 单应性矩阵声明
extern cv::Mat g_Homo_F; // 前视图单应性矩阵
extern cv::Mat g_Homo_B; // 后视图单应性矩阵
extern cv::Mat g_Homo_L; // 左视图单应性矩阵
extern cv::Mat g_Homo_R; // 右视图单应性矩阵

// 公共辅助函数声明
void rotate(std::string src_image_path, std::string dst_image_path, double angle1);
void init_des_points();
void init_params();

#endif // AVM_COMMON_H
