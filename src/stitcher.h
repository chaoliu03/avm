/**
 * @file stitcher.h
 * @brief 图像拼接与融合模块声明
 */

#ifndef AVM_STITCHER_H
#define AVM_STITCHER_H

#include "common.h"

/**
 * @brief 全景图拼接主函数
 * @description 将四个方向的鸟瞰图拼接成一张完整的全景图并叠加车辆模型
 */
void join();

// 辅助图像融合与叠加函数声明
cv::Mat ImageMerge(cv::Mat& src_image, cv::Mat& des_image, ImageType src_image_type);

cv::Mat ImageMergeWithMask(cv::Mat& src_image, cv::Mat& mask_image, cv::Mat& des_image, ImageType src_image_type);

/**
 * @brief 将四张去畸变图像按十字方位拼接并利用窗口显示出来
 * @param front 前向去畸变图像
 * @param back 后向去畸变图像
 * @param left 左侧去畸变图像
 * @param right 右侧去畸变图像
 */
void showUndistortStitched(const cv::Mat& front, const cv::Mat& back, const cv::Mat& left, const cv::Mat& right);

#endif // AVM_STITCHER_H
