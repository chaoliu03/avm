/**
 * @file stitcher.h
 * @brief 图像拼接与融合模块声明
 */

#ifndef AVM_STITCHER_H
#define AVM_STITCHER_H

#include "common.h"
// 图像交互缩放与拖拽状态结构体
struct PanZoomState
{
    cv::Mat     canvas;
    double      scale       = 1.0;
    cv::Point2d offset      = cv::Point2d(0, 0);
    bool        is_dragging = false;
    cv::Point   drag_start_mouse;
    cv::Point2d drag_start_offset;
    std::string window_name;

    void update_display()
    {
        // 获取窗口当前的实际图像显示区域大小，从而进行自适应分辨率的 warp 映射
        cv::Rect rect  = cv::getWindowImageRect(window_name);
        int      win_w = rect.width > 0 ? rect.width : 1000;
        int      win_h = rect.height > 0 ? rect.height : 1000;

        cv::Mat display_img;
        cv::Mat M = (cv::Mat_<double>(2, 3) << scale, 0, offset.x, 0, scale, offset.y);
        // 从原始高清大画布直接 warp 到窗口所占的分辨率大小，保证像素不被压缩，且缩放平移极致流畅
        cv::warpAffine(canvas, display_img, M, cv::Size(win_w, win_h), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(40, 40, 40));
        cv::imshow(window_name, display_img);
    }
};

/**
 * @brief 全景图拼接主函数
 * @description 将四个方向的鸟瞰图拼接成一张完整的全景图并叠加车辆模型
 */
void join();

/**
 * @brief 将四张去畸变图像按十字方位进行高清无损拼接，并返回大画布图像
 * @param front 前向去畸变图像
 * @param back 后向去畸变图像
 * @param left 左侧去畸变图像
 * @param right 右侧去畸变图像
 * @return 拼接后的高清大画布图像
 */
cv::Mat getUndistortStitched(const cv::Mat& front, const cv::Mat& back, const cv::Mat& left, const cv::Mat& right);

/**
 * @brief 采用自适应视口以及 Bicubic 插值算法对画布进行交互式缩放与拖拽配置
 * @param canvas 待显示的大画布图像
 * @param state 交互状态对象引用
 */
void showInteractive(const cv::Mat& canvas, PanZoomState& state);

#endif // AVM_STITCHER_H