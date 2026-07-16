/**
 * @file undistort.cpp
 * @brief 图像去畸变处理模块实现
 */

#include "undistort.h"
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

/**
 * @brief Undistort 类构造函数
 * @description 初始化鱼眼相机的各项参数和内参矩阵
 */
Undistort::Undistort()
{
    // 初始化相机参数
    m_fish_scale   = 0.5f;    // 鱼眼缩放因子
    m_focal_length = 910.0f;  // 焦距
    m_dx           = 3.0f;    // X 方向像素间距
    m_dy           = 3.0f;    // Y 方向像素间距
    m_fish_width   = 1280.0f; // 鱼眼图像宽度
    m_fish_height  = 960.0f;  // 鱼眼图像高度
    m_undis_scale  = 1.55f;   // 去畸变缩放因子

    // 去畸变到鱼眼的多项式参数
    m_undis2fish_params = {0.18238692, -0.08579553, 0.03366532, -0.00561911};

    // 计算去畸变后的图像尺寸
    m_undis_width  = static_cast<int>(m_undis_scale * m_fish_width);
    m_undis_height = static_cast<int>(m_undis_scale * m_fish_height);

    // 构建去畸变内参矩阵
    m_intrinsic_undis = (cv::Mat_<float>(3, 3) << m_focal_length / m_dx * m_fish_scale, 0, m_fish_width / 2 * m_undis_scale, 0, m_focal_length / m_dy * m_fish_scale, m_fish_height / 2 * m_undis_scale, 0, 0, 1);

    cout << "[INFO] Undistortion intrinsic matrix initialization completed" << endl;

    // 构建原始内参矩阵
    m_intrinsic = (cv::Mat_<float>(3, 3) << m_focal_length / m_dx, 0, m_fish_width / 2, 0, m_focal_length / m_dy, m_fish_height / 2, 0, 0, 1);

    cout << "[INFO] Original intrinsic matrix initialization completed" << endl;
}

/**
 * @brief OpenCV 风格的点坐标变换
 * @param warp_xy 输出变换后的坐标
 * @param map_center_h 映射中心 Y 坐标
 * @param map_center_w 映射中心 X 坐标
 * @param x_ 输入 X 坐标
 * @param y_ 输入 Y 坐标
 * @param scale 缩放因子
 */
void warpPointOpencv(cv::Vec2f& warp_xy, float map_center_h, float map_center_w, float x_, float y_, float scale)
{
    warp_xy[0] = x_ * scale + map_center_w;
    warp_xy[1] = y_ * scale + map_center_h;
}

/**
 * @brief 从去畸变坐标系变换到鱼眼坐标系
 * @description 使用多项式畸变模型实现坐标变换
 * @param fish_scale 鱼眼缩放因子
 * @param f_dx X 方向焦距与像素间距之比
 * @param f_dy Y 方向焦距与像素间距之比
 * @param large_center_h 去畸变图像中心 Y 坐标
 * @param large_center_w 去畸变图像中心 X 坐标
 * @param fish_center_h 鱼眼图像中心 Y 坐标
 * @param fish_center_w 鱼眼图像中心 X 坐标
 * @param undis_param 去畸变参数
 * @param x 输入 X 坐标
 * @param y 输入 Y 坐标
 * @return 变换后的鱼眼坐标
 */
cv::Vec2f warpUndist2Fisheye(float fish_scale, float f_dx, float f_dy, float large_center_h, float large_center_w, float fish_center_h, float fish_center_w, const cv::Vec4d& undis_param, float x, float y)
{
    f_dx *= fish_scale;
    f_dy *= fish_scale;

    // 转换至归一化成像平面坐标
    float y_ = (y - large_center_h) / f_dy;
    float x_ = (x - large_center_w) / f_dx;
    float r_ = static_cast<float>(sqrt(pow(x_, 2) + pow(y_, 2)));

    // 计算入射角
    float angle_undistorted = atan(r_);

    // 多项式展开
    float angle_undistorted_p2 = angle_undistorted * angle_undistorted;
    float angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted;
    float angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3;
    float angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5;
    float angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7;

    // 应用畸变模型
    float angle_distorted = static_cast<float>(
        angle_undistorted + undis_param[0] * angle_undistorted_p3 +
        undis_param[1] * angle_undistorted_p5 +
        undis_param[2] * angle_undistorted_p7 +
        undis_param[3] * angle_undistorted_p9);

    // 计算缩放比例
    float     scale = angle_distorted / (r_ + 0.00001f);
    cv::Vec2f warp_xy;

    float xx = (x - large_center_w) / fish_scale;
    float yy = (y - large_center_h) / fish_scale;

    warpPointOpencv(warp_xy, fish_center_h, fish_center_w, xx, yy, scale);

    return warp_xy;
}

void Undistort::getUndistortMap(vector<cv::Mat>& remap_table, int undist_w, int undist_h, cv::Mat intrinsic_undis, cv::Mat intrinsic_fish, cv::Vec4d undis_param, float fish_scale)
{
    float fisheye_width  = intrinsic_fish.at<float>(0, 2) * 2.0f;
    float fisheye_height = intrinsic_fish.at<float>(1, 2) * 2.0f;

    cv::Mat map_x(undist_h, undist_w, CV_32F);
    cv::Mat map_y(undist_h, undist_w, CV_32F);

    // 计算每个像素的重映射坐标
    for (int i = 0; i < undist_h; i++)
    {
        float* row_x = map_x.ptr<float>(i);
        float* row_y = map_y.ptr<float>(i);

        for (int j = 0; j < undist_w; j++)
        {
            cv::Vec2f xy = warpUndist2Fisheye(
                fish_scale, intrinsic_fish.at<float>(0, 0), intrinsic_fish.at<float>(1, 1), intrinsic_undis.at<float>(1, 2), intrinsic_undis.at<float>(0, 2), intrinsic_fish.at<float>(1, 2), intrinsic_fish.at<float>(0, 2), undis_param, static_cast<float>(j), static_cast<float>(i));

            // 边界保护
            xy[0] = xy[0] >= 0 ? xy[0] : 0.0f;
            xy[1] = xy[1] >= 0 ? xy[1] : 0.0f;
            xy[0] = xy[0] < fisheye_width ? xy[0] : fisheye_width - 1.0f;
            xy[1] = xy[1] < fisheye_height ? xy[1] : fisheye_height - 1.0f;

            row_x[j] = xy[0];
            row_y[j] = xy[1];
        }
    }

    remap_table.push_back(map_x);
    remap_table.push_back(map_y);
}

cv::Mat Undistort::undistort_func(cv::Mat img, vector<cv::Mat>& remap_table)
{
    // 标定初始化
    getUndistortMap(remap_table, m_undis_width, m_undis_height, m_intrinsic_undis, m_intrinsic, m_undis2fish_params, m_fish_scale);

    cv::Mat undis_img;
    cv::remap(img, undis_img, remap_table[0], remap_table[1], cv::INTER_LINEAR);
    return undis_img;
}
