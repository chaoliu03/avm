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

// 输出图片文件夹常量
const std::string OUTPUT_DIR = "output";

// 相机配置参数结构体
struct CameraConfig
{
    float     fish_scale;        // 鱼眼缩放因子
    float     focal_length;      // 焦距
    float     dx;                // X 方向像素间距
    float     dy;                // Y 方向像素间距
    int       fish_width;        // 鱼眼图像宽度
    int       fish_height;       // 鱼眼图像高度
    float     undis_scale;       // 去畸变缩放因子
    cv::Vec4d fish2undis_params; // 鱼眼到去畸变多项式参数
    cv::Vec4d undis2fish_params; // 去畸变到鱼眼多项式参数

    // 默认构造函数
    CameraConfig()
        : fish_scale(0.0f)
        , focal_length(0.0f)
        , dx(1.0f)
        , dy(1.0f)
        , fish_width(0)
        , fish_height(0)
        , undis_scale(1.0f)
        , fish2undis_params(0, 0, 0, 0)
        , undis2fish_params(0, 0, 0, 0)
    {}

    // 参数化构造函数
    CameraConfig(float fs, float fl, float dx_, float dy_, int fw, int fh, float us, cv::Vec4d f2u, cv::Vec4d u2f)
        : fish_scale(fs)
        , focal_length(fl)
        , dx(dx_)
        , dy(dy_)
        , fish_width(fw)
        , fish_height(fh)
        , fish2undis_params(f2u)
        , undis2fish_params(u2f)
    {
        // 动态自适应计算去畸变缩放因子，实现图像处理无损（不压缩像素）
        if (fs > 0.0f)
        {
            undis_scale = 1.55f * (0.5f / fs);
        }
        else
        {
            undis_scale = us;
        }
    }

    // 计算去畸变内参矩阵
    cv::Mat getIntrinsicUndis() const
    {
        return (cv::Mat_<float>(3, 3) << focal_length / dx * fish_scale, 0, fish_width / 2.0f * undis_scale, 0, focal_length / dy * fish_scale, fish_height / 2.0f * undis_scale, 0, 0, 1.0f);
    }

    // 计算原始内参矩阵
    cv::Mat getIntrinsic() const
    {
        return (cv::Mat_<float>(3, 3) << focal_length / dx, 0, fish_width / 2.0f, 0, focal_length / dy, fish_height / 2.0f, 0, 0, 1.0f);
    }
};

// 全局默认相机配置声明
extern const CameraConfig g_default_camera_config;

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
