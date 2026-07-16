/**
 * @file undistort.h
 * @brief 图像去畸变处理模块声明
 */

#ifndef AVM_UNDISTORT_H
#define AVM_UNDISTORT_H

#include "common.h"

/**
 * @brief 图像去畸变处理类
 * @description 负责将鱼眼图像转换为去畸变图像
 */
class Undistort
{
public:
    /**
     * @brief 构造函数，初始化去畸变参数
     */
    Undistort();

    /**
     * @brief 执行图像去畸变
     * @param img 输入鱼眼图像
     * @param remap_table 重映射表（输出参数）
     * @return 去畸变图像
     */
    cv::Mat undistort_func(cv::Mat img, std::vector<cv::Mat>& remap_table);

private:
    /**
     * @brief 生成去畸变重映射表
     * @param remap_table 重映射表（输出）
     * @param undist_w 去畸变图像宽度
     * @param undist_h 去畸变图像高度
     * @param intrinsic_undis 去畸变内参矩阵
     * @param intrinsic_fish 鱼眼内参矩阵
     * @param undis_param 去畸变参数
     * @param fish_scale 鱼眼缩放因子
     */
    void getUndistortMap(std::vector<cv::Mat>& remap_table, int undist_w, int undist_h, cv::Mat intrinsic_undis, cv::Mat intrinsic_fish, cv::Vec4d undis_param, float fish_scale);

    // 成员变量
    int       m_undis_width;       // 去畸变图像宽度
    int       m_undis_height;      // 去畸变图像高度
    float     m_fish_scale;        // 鱼眼缩放因子
    float     m_focal_length;      // 焦距
    float     m_dx;                // X 方向像素间距
    float     m_dy;                // Y 方向像素间距
    float     m_fish_width;        // 鱼眼图像宽度
    float     m_fish_height;       // 鱼眼图像高度
    float     m_undis_scale;       // 去畸变缩放因子
    cv::Vec4d m_undis2fish_params; // 去畸变到鱼眼的多项式参数
    cv::Mat   m_intrinsic_undis;   // 去畸变内参矩阵
    cv::Mat   m_intrinsic;         // 原始内参矩阵
};

// 辅助坐标映射函数声明
void warpPointOpencv(cv::Vec2f& warp_xy, float map_center_h, float map_center_w, float x_, float y_, float scale);

cv::Vec2f warpUndist2Fisheye(float fish_scale, float f_dx, float f_dy, float large_center_h, float large_center_w, float fish_center_h, float fish_center_w, const cv::Vec4d& undis_param, float x, float y);

#endif // AVM_UNDISTORT_H
