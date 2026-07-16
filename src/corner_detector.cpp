/**
 * @file corner_detector.cpp
 * @brief 标定板角点检测模块实现
 */

#include "corner_detector.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

// ========================================
// 直方图处理相关私有辅助函数（基于 OpenCV 源码）
// ========================================

template <typename ArrayContainer>
static void icvGradientOfHistogram256(const ArrayContainer& piHist,
    ArrayContainer&                                         piHistGrad)
{
    CV_DbgAssert(piHist.size() == 256);
    CV_DbgAssert(piHistGrad.size() == 256);

    piHistGrad[0] = 0;
    int prev_grad = 0;

    for (int i = 1; i < 255; ++i)
    {
        int grad = piHist[i - 1] - piHist[i + 1];
        if (std::abs(grad) < 100)
        {
            if (prev_grad == 0)
                grad = -100;
            else
                grad = prev_grad;
        }
        piHistGrad[i] = grad;
        prev_grad     = grad;
    }
    piHistGrad[255] = 0;
}

template <int iWidth_, typename ArrayContainer>
static void icvSmoothHistogram256(const ArrayContainer& piHist,
    ArrayContainer&                                     piHistSmooth,
    int                                                 iWidth = 0)
{
    CV_DbgAssert(iWidth_ == 0 || (iWidth == iWidth_ || iWidth == 0));
    iWidth = (iWidth_ != 0) ? iWidth_ : iWidth;
    CV_Assert(iWidth > 0);
    CV_DbgAssert(piHist.size() == 256);
    CV_DbgAssert(piHistSmooth.size() == 256);

    for (int i = 0; i < 256; ++i)
    {
        int iIdx_min = std::max(0, i - iWidth);
        int iIdx_max = std::min(255, i + iWidth);
        int iSmooth  = 0;

        for (int iIdx = iIdx_min; iIdx <= iIdx_max; ++iIdx)
        {
            CV_DbgAssert(iIdx >= 0 && iIdx < 256);
            iSmooth += piHist[iIdx];
        }
        piHistSmooth[i] = iSmooth / (2 * iWidth + 1);
    }
}

template <typename ArrayContainer>
static void icvGetIntensityHistogram256(const cv::Mat& img,
    ArrayContainer&                                    piHist)
{
    // 初始化直方图
    for (int i = 0; i < 256; i++)
        piHist[i] = 0;

    // 统计每个像素值的出现次数
    for (int j = 0; j < img.rows; ++j)
    {
        const uchar* row = img.ptr<uchar>(j);
        for (int i = 0; i < img.cols; i++)
        {
            piHist[row[i]]++;
        }
    }
}

static cv::Mat icvBinarizationHistogramBased(cv::Mat img, int fish_undis_flag)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);

    int       iCols    = img.cols;
    int       iRows    = img.rows;
    int       iMaxPix  = iCols * iRows;
    int       iMaxPix1 = iMaxPix / 100;
    const int iNumBins = 256;
    const int iMaxPos  = 20;

    // 分配直方图缓冲区
    cv::AutoBuffer<int, 256> piHistIntensity(iNumBins);
    cv::AutoBuffer<int, 256> piHistSmooth(iNumBins);
    cv::AutoBuffer<int, 256> piHistGrad(iNumBins);
    cv::AutoBuffer<int>      piMaxPos(iMaxPos);

    // 计算灰度直方图
    icvGetIntensityHistogram256(img, piHistIntensity);

    // 平滑直方图分布
    icvSmoothHistogram256<1>(piHistIntensity, piHistSmooth);

    // 计算梯度
    icvGradientOfHistogram256(piHistSmooth, piHistGrad);

    // 检测零点（波峰）
    unsigned iCntMaxima = 0;
    for (int i = iNumBins - 2; (i > 2) && (iCntMaxima < iMaxPos); --i)
    {
        if ((piHistGrad[i - 1] < 0) && (piHistGrad[i] > 0))
        {
            int iSumAroundMax = piHistSmooth[i - 1] + piHistSmooth[i] + piHistSmooth[i + 1];
            if (!(iSumAroundMax < iMaxPix1 && i < 64))
            {
                piMaxPos[iCntMaxima++] = i;
            }
        }
    }

    int iThresh = 0;
    CV_Assert((size_t)iCntMaxima <= piMaxPos.size());

    if (iCntMaxima == 0)
    {
        // 未检测到波峰，使用中值灰度
        const int iMaxPix2 = iMaxPix / 2;
        for (int sum = 0, i = 0; i < 256; ++i)
        {
            sum += piHistIntensity[i];
            if (sum > iMaxPix2)
            {
                iThresh = i;
                break;
            }
        }
    }
    else if (iCntMaxima == 1)
    {
        // 单峰分布
        iThresh = piMaxPos[0] / 2;
    }
    else if (iCntMaxima == 2)
    {
        // 双峰分布
        iThresh = (piMaxPos[0] + piMaxPos[1]) / 2;
    }
    else
    { // iCntMaxima >= 3, 多峰分布
        // 检查白区的阈值
        int iIdxAccSum = 0, iAccum = 0;
        for (int i = iNumBins - 1; i > 0; --i)
        {
            iAccum += piHistIntensity[i];
            if (iAccum > (iMaxPix / 5))
            {
                iIdxAccSum = i;
                break;
            }
        }

        unsigned iIdxBGMax  = 0;
        int      iBrightMax = piMaxPos[0];

        // 寻找最接近白区的零点
        for (unsigned n = 0; n < iCntMaxima - 1; ++n)
        {
            iIdxBGMax = n + 1;
            if (piMaxPos[n] < iIdxAccSum)
            {
                break;
            }
            iBrightMax = piMaxPos[n];
        }

        // 检查黑区的阈值
        int iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];

        // 如果太接近 255，跳到下一个波峰
        if (piMaxPos[iIdxBGMax] >= 250 && iIdxBGMax + 1 < iCntMaxima)
        {
            iIdxBGMax++;
            iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];
        }

        // 寻找最大的黑色波峰
        for (unsigned n = iIdxBGMax + 1; n < iCntMaxima; n++)
        {
            if (piHistIntensity[piMaxPos[n]] >= iMaxVal)
            {
                iMaxVal   = piHistIntensity[piMaxPos[n]];
                iIdxBGMax = n;
            }
        }

        // 设置二值化阈值
        int iDist2 = (iBrightMax - piMaxPos[iIdxBGMax]) / 2;
        iThresh    = iBrightMax - iDist2;

        // 针对鱼眼暗区的特殊处理
        if (fish_undis_flag == 0)
        {
            auto temp = static_cast<float>(iThresh) * 0.8f;
            iThresh   = static_cast<int>(temp);
        }
    }

    cv::Mat img_thresh(img.rows, img.cols, img.type());
    if (iThresh > 0)
    {
        img_thresh = (img >= iThresh);
    }
    return img_thresh;
}

// ========================================
// 轮廓排序辅助比较函数
// ========================================

static bool cmp(std::vector<cv::Point> A, std::vector<cv::Point> B)
{
    return (contourArea(A) > contourArea(B));
}

static bool cmpYmax(cv::Point A, cv::Point B)
{
    return (A.y > B.y);
}

static bool cmpXmax(cv::Point A, cv::Point B)
{
    return (A.x > B.x);
}

static bool cmpYmin(cv::Point A, cv::Point B)
{
    return (A.y < B.y);
}

static bool cmpXmin(cv::Point A, cv::Point B)
{
    return (A.x < B.x);
}

/**
 * @brief 对检测到的角点进行排序
 * @description 将 8 个角点按从上到下、从左到右的顺序进行排序
 * @param points 输入的 8 个角点
 * @return 排序后的角点序列
 */
static std::vector<cv::Point2f> detectPointsSort(std::vector<cv::Point2f> points)
{
    // 按 Y 坐标从小到大排序
    sort(points.begin(), points.end(), cmpYmin);

    // 分为两行，每行 4 个点
    std::vector<cv::Point2f> mid1(points.begin(), points.begin() + 4);
    std::vector<cv::Point2f> mid2(points.begin() + 4, points.begin() + 8);

    // 在每一行内按 X 坐标从小到大排序
    sort(mid1.begin(), mid1.end(), cmpXmin);
    sort(mid2.begin(), mid2.end(), cmpXmin);

    // 合并结果
    std::vector<cv::Point2f> sortedPoints(mid1.begin(), mid1.end());
    sortedPoints.insert(sortedPoints.end(), mid2.begin(), mid2.end());

    return sortedPoints;
}

// ========================================
// 核心函数实现
// ========================================

/**
 * @brief 角点检测的图像对比度增强
 * @description 通过 Gamma 矫正增强图像对比度，以便后续进行角点检测
 * @param img 输入三通道图像
 * @param contrast 对比度增强系数
 * @return 增强后的图像
 */
cv::Mat imgAugForPointDetect(const cv::Mat img, float contrast)
{
    cv::Mat mat_float_tmp;

    // 转换为 32 位浮点型三通道图像
    img.convertTo(mat_float_tmp, CV_32FC3);

    // 归一化到 [0,1]
    mat_float_tmp = mat_float_tmp / 255.0f;

    // 寻找最大值
    float maxvalue_ = 0.0f;
    for (int i = 0; i < mat_float_tmp.rows; ++i)
    {
        float* data = mat_float_tmp.ptr<float>(i);
        for (int j = 0; j < mat_float_tmp.cols; ++j)
        {
            if (data[j] > maxvalue_)
            {
                maxvalue_ = data[j];
            }
        }
    }

    // 再次归一化
    mat_float_tmp = mat_float_tmp / maxvalue_;

    // 应用 Gamma 矫正增强对比度
    pow(mat_float_tmp, contrast, mat_float_tmp);
    mat_float_tmp = mat_float_tmp * 255.0f;

    // 转换回 8 位无符号整数
    cv::Mat contrast_img;
    mat_float_tmp.convertTo(contrast_img, CV_8UC3);

    return contrast_img;
}

/**
 * @brief 在二值图像中寻找矩形（标定板角点）
 * @description 通过轮廓检测和多边形逼近寻找四边形标定板
 * @param img 输入二值图像
 * @param valid_region_y Y 方向有效检测区域
 * @param max_sz 最大面积阈值
 * @param fish_scale 鱼眼缩放因子
 * @param detect_points 检测到的角点坐标（输出）
 * @param fish_undis_flag 鱼眼去畸变标志
 */
void findRectangle(cv::Mat img, std::vector<float> valid_region_y, float max_sz, float fish_scale, std::vector<cv::Point2f>& detect_points, int fish_undis_flag)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;

    float min_y_thresh = valid_region_y[0];
    float max_y_thresh = valid_region_y[1];

    // 计算最小面积阈值
    int min_size = static_cast<int>(max_sz * (pow(fish_scale, 2)));
    if (fish_undis_flag == 0)
    {
        auto temp = static_cast<float>(min_size) * 0.5f;
        min_size  = static_cast<int>(temp);
    }

    float approx_level = 10.0f * fish_scale;

    // 提取轮廓
    cv::findContours(img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect  contour_rect;
    cv::Point pt[4];

    for (int idx = (int)(contours.size() - 1); idx >= 0; --idx)
    {
        auto contour = contours[idx];
        contour_rect = boundingRect(contour);

        // 面积过滤
        if (contour_rect.area() < min_size)
        {
            continue;
        }

        std::vector<cv::Point> approx_contour;
        // 多边形曲线逼近
        cv::approxPolyDP(contour, approx_contour, approx_level, true);

        // 检查是否为四边形
        if (approx_contour.size() == 4)
        {
            for (int i = 0; i < 4; ++i)
                pt[i] = approx_contour[i];

            // 检查中心点是否在内部（黑色）
            int x_lable = (pt[0].x + pt[2].x) / 2;
            int y_lable = (pt[0].y + pt[2].y) / 2;
            if (img.at<uchar>(y_lable, x_lable) != 0)
            {
                continue;
            }

            // 计算周长和面积
            double p    = cv::arcLength(approx_contour, true);
            double area = cv::contourArea(approx_contour, false);

            // 计算对角线长度
            double d1 = sqrt(cv::normL2Sqr<double>(pt[0] - pt[2]));
            double d2 = sqrt(cv::normL2Sqr<double>(pt[1] - pt[3]));

            // 计算边长
            double d3 = sqrt(cv::normL2Sqr<double>(pt[0] - pt[1]));
            double d4 = sqrt(cv::normL2Sqr<double>(pt[1] - pt[2]));

            // 形状验证：检查是否接近正方形
            if (!(d3 * 5 > d4 && d4 * 5 > d3 && d3 * d4 < area * 10 &&
                    area > min_size && d1 >= 0.1 * p && d2 >= 0.1 * p))
            {
                continue;
            }

            // Y 坐标区域验证
            float contour_y_average = static_cast<float>(
                                          approx_contour[0].y + approx_contour[1].y +
                                          approx_contour[2].y + approx_contour[3].y) /
                4.0f;

            if (contour_y_average < min_y_thresh || contour_y_average > max_y_thresh)
            {
                continue;
            }

            // 添加检测到的角点
            detect_points.push_back(approx_contour[0]);
            detect_points.push_back(approx_contour[1]);
            detect_points.push_back(approx_contour[2]);
            detect_points.push_back(approx_contour[3]);
        }
    }
}

/**
 * @brief 点坐标反变换
 * @param warp_xy 输出变换后的坐标
 * @param map_center_h 映射中心 Y 坐标
 * @param map_center_w 映射中心 X 坐标
 * @param x_ 输入 X 坐标
 * @param y_ 输入 Y 坐标
 * @param scale 缩放因子
 */
void warpPointInverse(cv::Vec2f& warp_xy, float map_center_h, float map_center_w, float x_, float y_, float scale)
{
    warp_xy[0] = x_ * scale + map_center_w;
    warp_xy[1] = y_ * scale + map_center_h;
}

/**
 * @brief 从鱼眼坐标系变换到去畸变坐标系
 * @description 实现从鱼眼图像到正常透视图像的坐标变换
 * @param fish_scale 鱼眼缩放因子
 * @param f_dx X 方向焦距与像素间距之比
 * @param f_dy Y 方向焦距与像素间距之比
 * @param undis_center_h 去畸变图像中心 Y 坐标
 * @param undis_center_w 去畸变图像中心 X 坐标
 * @param fish_center_h 鱼眼图像中心 Y 坐标
 * @param fish_center_w 鱼眼图像中心 X 坐标
 * @param undis_param 去畸变参数
 * @param x 输入 X 坐标
 * @param y 输入 Y 坐标
 * @return 变换后的去畸变坐标
 */
cv::Vec2f warpFisheye2Undist(float fish_scale, float f_dx, float f_dy, float undis_center_h, float undis_center_w, float fish_center_h, float fish_center_w, cv::Vec4d undis_param, float x, float y)
{
    // 像素投影到归一化成像坐标系
    float y_          = (y - fish_center_h) / f_dy;
    float x_          = (x - fish_center_w) / f_dx;
    float r_distorted = static_cast<float>(sqrt(pow(x_, 2) + pow(y_, 2)));

    // 根据畸变公式计算入射角
    float r_distorted_p2 = r_distorted * r_distorted;
    float r_distorted_p3 = r_distorted_p2 * r_distorted;
    float r_distorted_p4 = r_distorted_p2 * r_distorted_p2;
    float r_distorted_p5 = r_distorted_p2 * r_distorted_p3;

    float angle_undistorted = static_cast<float>(
        r_distorted + undis_param[0] * r_distorted_p2 +
        undis_param[1] * r_distorted_p3 + undis_param[2] * r_distorted_p4 +
        undis_param[3] * r_distorted_p5);

    // 计算归一化图像中的缩放比例
    float r_undistorted = tanf(angle_undistorted);
    float scale         = r_undistorted / (r_distorted + 0.00001f);

    cv::Vec2f warp_xy;

    // 转换到实际图像坐标系
    float xx = (x - fish_center_w) * fish_scale;
    float yy = (y - fish_center_h) * fish_scale;

    // 投影到去畸变图像坐标系
    warpPointInverse(warp_xy, undis_center_h, undis_center_w, xx, yy, scale);

    return warp_xy;
}

/**
 * @brief 检测标定板角点
 * @description 在图像中检测 2x4 排列的标定板角点，并转换到去畸变坐标系下
 * @param img 输入图像
 * @param max_sz 最大面积阈值
 * @param config 相机配置参数结构体
 * @param detect_points 检测到的角点坐标（输出）
 * @param fish_undis_flag 鱼眼去畸变标志
 * @param src_image_type 源图像类型
 * @return 是否成功检测到 8 个角点
 */
bool detectPoints(cv::Mat img, float max_sz, const CameraConfig& config, std::vector<cv::Point2f>& detect_points, int fish_undis_flag, ImageType src_image_type)
{
    float fish_scale = config.fish_scale;

    // 设置有效的检测区域（Y 方向）
    float              max_y_thresh = static_cast<float>(0.7f * img.rows);
    float              min_y_thresh = static_cast<float>(0.2f * img.rows);
    std::vector<float> y_valid_area{min_y_thresh, max_y_thresh};

    // 转换为灰度图像
    if (img.channels() != 1)
    {
        cv::cvtColor(img, img, COLOR_BGR2GRAY);
    }

    // 图像对比度增强
    float   contrast     = 2.5f;
    cv::Mat img_contrast = imgAugForPointDetect(img, contrast);

    // 保存增强后的图像用于调试
    switch (src_image_type)
    {
    case ImageType::IMAGE_FRONT:
        cv::imwrite("build/front_img_contrast.jpg", img_contrast);
        break;
    case ImageType::IMAGE_BACK:
        cv::imwrite("build/back_img_contrast.jpg", img_contrast);
        break;
    case ImageType::IMAGE_LEFT:
        cv::imwrite("build/left_img_contrast.jpg", img_contrast);
        break;
    case ImageType::IMAGE_RIGHT:
        cv::imwrite("build/right_img_contrast.jpg", img_contrast);
        break;
    default:
        cv::imwrite("build/img_contrast.jpg", img_contrast);
    }

    // 基于直方图的二值化
    cv::Mat img_thresh = icvBinarizationHistogramBased(img_contrast, 0);

    // 寻找矩形角点
    findRectangle(img_thresh, y_valid_area, max_sz, fish_scale, detect_points, fish_undis_flag);

    // 如果未找到 8 个角点，使用备用方案
    if (detect_points.size() != 8)
    {
        // 自适应阈值二值化
        cv::adaptiveThreshold(img_contrast, img_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 401, 5);
        detect_points.clear();
        findRectangle(img_thresh, y_valid_area, max_sz, fish_scale, detect_points, fish_undis_flag);
    }

    // 在二值图像上绘制检测到的角点用于调试
    for (int i = 0; i < detect_points.size(); i++)
    {
        cv::circle(img_thresh, detect_points[i], 1, cv::Scalar(0, 255, 0), 5);
    }

    // 保存二值化结果
    switch (src_image_type)
    {
    case ImageType::IMAGE_FRONT:
        cv::imwrite("build/front_img_thresh.jpg", img_thresh);
        break;
    case ImageType::IMAGE_BACK:
        cv::imwrite("build/back_img_thresh.jpg", img_thresh);
        break;
    case ImageType::IMAGE_LEFT:
        cv::imwrite("build/left_img_thresh.jpg", img_thresh);
        break;
    case ImageType::IMAGE_RIGHT:
        cv::imwrite("build/right_img_thresh.jpg", img_thresh);
        break;
    default:
        cv::imwrite("build/img_thresh.jpg", img_thresh);
    }

    // 检查是否成功检测到 8 个角点
    if (detect_points.size() != 8)
    {
        return false;
    }

    // 亚像素级角点精细化
    cv::cornerSubPix(img, detect_points, cv::Size(9, 9), cv::Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

    // 对角点排序
    detect_points = detectPointsSort(detect_points);

    // 获取相机内参和畸变参数
    float     f_dx              = config.focal_length / config.dx;
    float     f_dy              = config.focal_length / config.dy;
    float     fish_center_x     = static_cast<float>(config.fish_width) / 2.0f;
    float     fish_center_y     = static_cast<float>(config.fish_height) / 2.0f;
    float     undis_center_x    = (static_cast<float>(config.fish_width) / 2.0f) * config.undis_scale;
    float     undis_center_y    = (static_cast<float>(config.fish_height) / 2.0f) * config.undis_scale;
    cv::Vec4d fish2undis_params = config.fish2undis_params;

    // 将角点从鱼眼坐标系转换到去畸变坐标系
    cv::Vec2f xy;
    for (int j = 0; j < 8; j++)
    {
        xy = warpFisheye2Undist(fish_scale, f_dx, f_dy, undis_center_y, undis_center_x, fish_center_y, fish_center_x, fish2undis_params, detect_points[j].x, detect_points[j].y);

        detect_points[j].x = xy[0];
        detect_points[j].y = xy[1];
    }

    return true;
}
