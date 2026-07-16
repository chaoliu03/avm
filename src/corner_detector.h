/**
 * @file corner_detector.h
 * @brief 标定板角点检测模块声明
 */

#ifndef AVM_CORNER_DETECTOR_H
#define AVM_CORNER_DETECTOR_H

#include "common.h"

/**
 * @brief 检测标定板角点
 * @description 在图像中检测 2x4 排列的标定板角点，并转换到去畸变坐标系下
 * @param img 输入图像
 * @param max_sz 最大面积分数
 * @param config 相机配置参数结构体
 * @param detect_points 检测到的角点坐标（输出）
 * @param fish_undis_flag 鱼眼去畸变标志
 * @param src_image_type 源图像类型
 * @return 是否成功检测到 8 个角点
 */
bool detectPoints(cv::Mat img, float max_sz, const CameraConfig& config, std::vector<cv::Point2f>& detect_points, int fish_undis_flag, ImageType src_image_type);

// 辅助检测与图像增强函数声明
cv::Mat imgAugForPointDetect(const cv::Mat img, float contrast);

void findRectangle(cv::Mat img, std::vector<float> valid_region_y, float max_sz, float fish_scale, std::vector<cv::Point2f>& detect_points, int fish_undis_flag);

void warpPointInverse(cv::Vec2f& warp_xy, float map_center_h, float map_center_w, float x_, float y_, float scale);

cv::Vec2f warpFisheye2Undist(float fish_scale, float f_dx, float f_dy, float undis_center_h, float undis_center_w, float fish_center_h, float fish_center_w, cv::Vec4d undis_param, float x, float y);

#endif // AVM_CORNER_DETECTOR_H
