/**
 * @file avm.cpp
 * @brief 环视全景 (AVM) 系统实现
 * @description 本程序实现了一个基于四个鱼眼相机的环视系统，
 *              包括图像去畸变、角点检测、透视变换和图像拼接。
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <sys/stat.h> // 用于创建目录
#ifdef _WIN32
#include <direct.h> // 用于 Windows 上的 _mkdir
#endif

using namespace cv;
using namespace std;

// 常量定义
#define IMAGE_BACK_PIXEL_Y  643 // 拼接结果中后视图图像的 Y 轴偏移量
#define IMAGE_RIGHT_PIXEL_X 398 // 拼接结果中右视图图像的 X 轴偏移量

// 全局变量
cv::Mat   g_intrinsic_undis;   // 去畸变内参矩阵
cv::Mat   g_intrinsic;         // 原始内参矩阵
cv::Vec4d g_fish2undis_params; // 鱼眼到无畸变变换参数

// 四个方向的角点坐标
std::vector<cv::Point2f> g_corner_front; // 前视图角点
std::vector<cv::Point2f> g_corner_back;  // 后视图角点
std::vector<cv::Point2f> g_corner_left;  // 左视图角点
std::vector<cv::Point2f> g_corner_right; // 右视图角点

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
    cv::Mat undistort_func(cv::Mat img, vector<cv::Mat>& remap_table);

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
    void getUndistortMap(vector<cv::Mat>& remap_table, int undist_w, int undist_h, cv::Mat intrinsic_undis, cv::Mat intrinsic_fish, cv::Vec4d undis_param, float fish_scale);

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

// ========================================
// 直方图处理相关函数（基于 OpenCV 源码）
// ========================================

/**
 * @brief 计算 256 级直方图的梯度
 * @description 用于检测直方图中的波峰和波谷
 * @param piHist 输入直方图
 * @param piHistGrad 输出梯度直方图
 */
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

/**
 * @brief 使用滑动窗口平滑直方图
 * @description 减少直方图中的噪声，窗口大小为 2*iWidth+1
 * @param piHist 输入直方图
 * @param piHistSmooth 输出平滑后的直方图
 * @param iWidth 平滑窗口半径
 */
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

/**
 * @brief 计算图像的灰度直方图
 * @description 统计每个灰度级的像素个数
 * @param img 输入单通道图像
 * @param piHist 输出 256 级直方图
 */
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

/**
 * @brief 基于双峰直方图的图像二值化
 * @description 通过分析图像直方图自动确定最佳阈值，将图像分割为前景和背景
 * @param img 输入单通道图像
 * @param fish_undis_flag 是否为鱼眼去畸变标志
 * @return 二值化图像
 */
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
// 图像处理与角点检测相关函数
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

// ========================================
// 排序比较函数
// ========================================

/**
 * @brief 按面积降序排序轮廓的比较函数
 */
bool cmp(std::vector<cv::Point> A, std::vector<cv::Point> B)
{
    return (contourArea(A) > contourArea(B));
}

/**
 * @brief 按 Y 坐标降序排序的比较函数
 */
bool cmpYmax(cv::Point A, cv::Point B)
{
    return (A.y > B.y);
}

/**
 * @brief 按 X 坐标降序排序的比较函数
 */
bool cmpXmax(cv::Point A, cv::Point B)
{
    return (A.x > B.x);
}

/**
 * @brief 按 Y 坐标升序排序的比较函数
 */
bool cmpYmin(cv::Point A, cv::Point B)
{
    return (A.y < B.y);
}

/**
 * @brief 按 X 坐标升序排序的比较函数
 */
bool cmpXmin(cv::Point A, cv::Point B)
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
// 坐标变换相关函数
// ========================================

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
 * @param fish_scale 鱼眼缩放因子
 * @param detect_points 检测到的角点坐标（输出）
 * @param fish_undis_flag 鱼眼去畸变标志
 * @param src_image_type 源图像类型
 * @return 是否成功检测到 8 个角点
 */
bool detectPoints(cv::Mat img, float max_sz, float fish_scale, std::vector<cv::Point2f>& detect_points, int fish_undis_flag, ImageType src_image_type)
{
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

    // 获取相机内参参数
    float     f_dx              = g_intrinsic.at<float>(0, 0);
    float     f_dy              = g_intrinsic.at<float>(1, 1);
    float     fish_center_x     = g_intrinsic.at<float>(0, 2);
    float     fish_center_y     = g_intrinsic.at<float>(1, 2);
    float     undis_center_x    = g_intrinsic_undis.at<float>(0, 2);
    float     undis_center_y    = g_intrinsic_undis.at<float>(1, 2);
    cv::Vec4d fish2undis_params = g_fish2undis_params;

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

// ========================================
// 鸟瞰图变换相关全局变量与函数
// ========================================

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
 * @brief 初始化前视图鸟瞰目标角点
 * @description 设置鸟瞰图中前视图图像的标准角点位置
 */
void init_des_front_points(void)
{
    g_corner_bird_front = std::vector<cv::Point2i>(8);

    // 第一行角点 X 坐标
    g_corner_bird_front[0].x = 136;
    g_corner_bird_front[1].x = 256;
    g_corner_bird_front[2].x = 536;
    g_corner_bird_front[3].x = 656;

    // 第二行角点 X 坐标
    g_corner_bird_front[4].x = 136;
    g_corner_bird_front[5].x = 256;
    g_corner_bird_front[6].x = 536;
    g_corner_bird_front[7].x = 656;

    // 第一行角点 Y 坐标
    g_corner_bird_front[0].y = 85;
    g_corner_bird_front[1].y = 85;
    g_corner_bird_front[2].y = 85;
    g_corner_bird_front[3].y = 85;

    // 第二行角点 Y 坐标
    g_corner_bird_front[4].y = 205;
    g_corner_bird_front[5].y = 205;
    g_corner_bird_front[6].y = 205;
    g_corner_bird_front[7].y = 205;
}

/**
 * @brief 初始化后视图鸟瞰目标角点
 * @description 设置鸟瞰图中后视图图像的标准角点位置
 */
void init_des_back_points(void)
{
    g_corner_bird_back = std::vector<cv::Point2i>(8);

    // 角点 X 坐标设置
    g_corner_bird_back[0].x = 136;
    g_corner_bird_back[1].x = 256;
    g_corner_bird_back[2].x = 536;
    g_corner_bird_back[3].x = 656;
    g_corner_bird_back[4].x = 136;
    g_corner_bird_back[5].x = 256;
    g_corner_bird_back[6].x = 536;
    g_corner_bird_back[7].x = 656;

    // 角点 Y 坐标设置
    g_corner_bird_back[0].y = 85;
    g_corner_bird_back[1].y = 85;
    g_corner_bird_back[2].y = 85;
    g_corner_bird_back[3].y = 85;
    g_corner_bird_back[4].y = 205;
    g_corner_bird_back[5].y = 205;
    g_corner_bird_back[6].y = 205;
    g_corner_bird_back[7].y = 205;
}

/**
 * @brief 初始化左视图鸟瞰目标角点
 * @description 设置鸟瞰图中左视图图像的标准角点位置
 */
void init_des_left_points(void)
{
    g_corner_bird_left = std::vector<cv::Point2i>(8);

    // 角点 X 坐标设置
    g_corner_bird_left[0].x = 85;
    g_corner_bird_left[1].x = 205;
    g_corner_bird_left[2].x = 926;
    g_corner_bird_left[3].x = 1046;
    g_corner_bird_left[4].x = 85;
    g_corner_bird_left[5].x = 205;
    g_corner_bird_left[6].x = 926;
    g_corner_bird_left[7].x = 1046;

    // 角点 Y 坐标设置
    g_corner_bird_left[0].y = 136;
    g_corner_bird_left[1].y = 136;
    g_corner_bird_left[2].y = 136;
    g_corner_bird_left[3].y = 136;
    g_corner_bird_left[4].y = 256;
    g_corner_bird_left[5].y = 256;
    g_corner_bird_left[6].y = 256;
    g_corner_bird_left[7].y = 256;
}

/**
 * @brief 初始化右视图鸟瞰目标角点
 * @description 设置鸟瞰图中右视图图像的标准角点位置
 */
void init_des_right_points(void)
{
    g_corner_bird_right = std::vector<cv::Point2i>(8);

    // 角点 X 坐标设置
    g_corner_bird_right[0].x = 85;
    g_corner_bird_right[1].x = 205;
    g_corner_bird_right[2].x = 926;
    g_corner_bird_right[3].x = 1046;
    g_corner_bird_right[4].x = 85;
    g_corner_bird_right[5].x = 205;
    g_corner_bird_right[6].x = 926;
    g_corner_bird_right[7].x = 1046;

    // 角点 Y 坐标设置
    g_corner_bird_right[0].y = 136;
    g_corner_bird_right[1].y = 136;
    g_corner_bird_right[2].y = 136;
    g_corner_bird_right[3].y = 136;
    g_corner_bird_right[4].y = 256;
    g_corner_bird_right[5].y = 256;
    g_corner_bird_right[6].y = 256;
    g_corner_bird_right[7].y = 256;
}

/**
 * @brief 初始化所有方向的鸟瞰目标角点
 * @description 调用各方向的角点初始化函数
 */
void init_des_points(void)
{
    init_des_front_points();
    init_des_back_points();
    init_des_left_points();
    init_des_right_points();
}

/**
 * @brief 初始化环视系统参数
 * @description 初始化所有必需的参数和数据结构
 */
void init_params(void)
{
    init_des_points();
}

// ========================================
// 图像旋转与拼接相关函数
// ========================================

/**
 * @brief 旋转图像
 * @description 将图像旋转指定角度并保存到目标路径
 * @param src_image_path 源图像路径
 * @param dst_image_path 目标图像路径
 * @param angle1 旋转角度（度）
 */
void rotate(string src_image_path, string dst_image_path, double angle1)
{ // 读取图像
    Mat image = imread(src_image_path);
    if (image.empty())
    {
        cout << "[ERROR] Unable to read image file: " << src_image_path << endl;
        return;
    }

    // 获取图像尺寸
    int height = image.rows;
    int width  = image.cols;

    // 设置旋转中心为图像中心
    Point2f center(width / 2.0f, height / 2.0f);

    // 设置旋转角度和缩放因子
    double angle = angle1;
    double scale = 1.0;

    // 获取旋转矩阵
    Mat rotationMatrix = getRotationMatrix2D(center, angle, scale);

    // 计算旋转后的图像尺寸以避免内容被裁剪
    Rect bbox = RotatedRect(center, image.size(), angle).boundingRect();

    // 调整旋转矩阵的平移部分以适应新的图像尺寸
    rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // 创建输出图像
    Mat rotatedImage;

    // 应用仿射变换（旋转）
    warpAffine(image, rotatedImage, rotationMatrix, bbox.size());

    // 保存旋转后的图像
    imwrite(dst_image_path, rotatedImage);
}

/**
 * @brief 图像拼接与合并
 * @description 将源图像合并到目标图像的指定位置
 * @param src_image 源图像
 * @param des_image 目标图像
 * @param src_image_type 源图像类型
 * @return 合并后的图像
 */
cv::Mat ImageMerge(cv::Mat& src_image, cv::Mat& des_image, ImageType src_image_type)
{
    if (src_image.empty() || des_image.empty())
    {
        return cv::Mat();
    }

    cv::Mat output;
    des_image.copyTo(output);

    src_image.convertTo(src_image, CV_32FC3);

    if (src_image_type == ImageType::IMAGE_FRONT || src_image_type == ImageType::IMAGE_BACK)
    {
        // 处理前视图和后视图图像
        for (int i = 0; i < src_image.rows; i++)
        {
            for (int j = 0; j < des_image.cols; j++)
            {
                for (int channel = 0; channel < 3; channel++)
                {
                    if (src_image_type == ImageType::IMAGE_FRONT)
                    {
                        output.at<cv::Vec3b>(i, j)[channel] =
                            static_cast<uchar>(src_image.at<cv::Vec3f>(i, j)[channel]);
                    }
                    else if (src_image_type == ImageType::IMAGE_BACK)
                    {
                        output.at<cv::Vec3b>(i + IMAGE_BACK_PIXEL_Y, j)[channel] =
                            static_cast<uchar>(src_image.at<cv::Vec3f>(i, j)[channel]);
                    }
                }
            }
        }
    }
    else if (src_image_type == ImageType::IMAGE_LEFT || src_image_type == ImageType::IMAGE_RIGHT)
    {
        // 处理左视图和右视图图像
        for (int i = 0; i < src_image.rows; i++)
        {
            for (int j = 0; j < src_image.cols; j++)
            {
                for (int channel = 0; channel < 3; channel++)
                {
                    if (src_image_type == ImageType::IMAGE_LEFT)
                    {
                        output.at<cv::Vec3b>(i, j)[channel] +=
                            static_cast<uchar>(src_image.at<cv::Vec3f>(i, j)[channel]);
                    }
                    else if (src_image_type == ImageType::IMAGE_RIGHT)
                    {
                        output.at<cv::Vec3b>(i, j + IMAGE_RIGHT_PIXEL_X)[channel] +=
                            static_cast<uchar>(src_image.at<cv::Vec3f>(i, j)[channel]);
                    }
                }
            }
        }
    }
    return output;
}

/**
 * @brief 带遮罩的图像拼接
 * @description 使用遮罩图像进行更精确的图像拼接
 * @param src_image 源图像
 * @param mask_image 遮罩图像
 * @param des_image 目标图像
 * @param src_image_type 源图像类型
 * @return 合并后的图像
 */
cv::Mat ImageMergeWithMask(cv::Mat& src_image, cv::Mat& mask_image, cv::Mat& des_image, ImageType src_image_type)
{
    cv::Mat output(des_image.rows, des_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    return output;
}

/**
 * @brief 全景图拼接主函数
 * @description 将四个方向的鸟瞰图拼接成一张完整的全景图并叠加车辆模型
 */
void join()
{
    // 读取四路鸟瞰图
    Mat img_front = imread("build/bird_front_2.jpg");
    Mat img_back  = imread("build/bird_back_2.jpg");
    Mat img_left  = imread("build/bird_left_2.jpg");
    Mat img_right = imread("build/bird_right_2.jpg");

    // 缩放图像以适应拼接
    cv::resize(img_front, img_front, cv::Size(616, 237));
    cv::resize(img_back, img_back, cv::Size(616, 237));
    cv::resize(img_left, img_left, cv::Size(218, 880));
    cv::resize(img_right, img_right, cv::Size(218, 880));

    // 读取遮罩图像（三通道）
    Mat mask_front = imread("assets/masks/maskFront.jpg");
    Mat mask_back  = imread("assets/masks/maskBack.jpg");
    Mat mask_left  = imread("assets/masks/maskLeft.jpg");
    Mat mask_right = imread("assets/masks/maskRight.jpg"); // 检查图像是否成功加载
    if (img_front.empty() || img_back.empty() || img_left.empty() || img_right.empty() ||
        mask_front.empty() || mask_back.empty() || mask_left.empty() || mask_right.empty())
    {
        cout << "[ERROR] Unable to load one or more image or mask files" << endl;
        return;
    }

    // 将遮罩图像归一化到 [0,1] 范围
    mask_front.convertTo(mask_front, CV_32FC3, 1.0 / 255.0);
    mask_back.convertTo(mask_back, CV_32FC3, 1.0 / 255.0);
    mask_left.convertTo(mask_left, CV_32FC3, 1.0 / 255.0);
    mask_right.convertTo(mask_right, CV_32FC3, 1.0 / 255.0);

    // 将原始图像转换为浮点型以便进行乘法运算
    Mat img_front_float, img_back_float, img_left_float, img_right_float;
    img_front.convertTo(img_front_float, CV_32FC3);
    img_back.convertTo(img_back_float, CV_32FC3);
    img_left.convertTo(img_left_float, CV_32FC3);
    img_right.convertTo(img_right_float, CV_32FC3);

    // 应用遮罩（逐像素相乘）
    Mat masked_front, masked_back, masked_left, masked_right;
    multiply(img_front_float, mask_front, masked_front);
    multiply(img_back_float, mask_back, masked_back);
    multiply(img_left_float, mask_left, masked_left);
    multiply(img_right_float, mask_right, masked_right);

    // 转换回 8 位无符号整数
    masked_front.convertTo(masked_front, CV_8UC3);
    masked_back.convertTo(masked_back, CV_8UC3);
    masked_left.convertTo(masked_left, CV_8UC3);
    masked_right.convertTo(masked_right, CV_8UC3);

    // 获取图像尺寸
    Size frontSize = masked_front.size();
    Size backSize  = masked_back.size();
    Size leftSize  = masked_left.size();
    Size rightSize = masked_right.size();

    // 计算拼接后图像的总尺寸
    int totalWidth  = frontSize.width;
    int totalHeight = leftSize.height;

    // 创建结果图像
    Mat result = Mat::zeros(totalHeight, totalWidth, CV_8UC3);

    // 拼接图像 - 左侧图像
    masked_left.copyTo(result(Rect(0, 0, leftSize.width, leftSize.height)));

    // 拼接图像 - 右侧图像（与左侧图像重叠区相加）
    Mat roi_right = result(Rect(totalWidth - rightSize.width, 0, rightSize.width, rightSize.height));
    add(roi_right, masked_right, roi_right);

    // 拼接图像 - 前侧图像（加到已有内容中）
    Mat roi_front = result(Rect(0, 0, frontSize.width, frontSize.height));
    add(roi_front, masked_front, roi_front);

    // 拼接图像 - 后侧图像（加到已有内容中）
    Mat roi_back = result(Rect(0, leftSize.height - backSize.height, backSize.width, backSize.height));
    add(roi_back, masked_back, roi_back);

    // 叠加车辆模型图像
    Mat img_su7 = imread("assets/images/su7.png", IMREAD_UNCHANGED);
    if (img_su7.empty())
    {
        cout << "[ERROR] Unable to load vehicle model image file: assets/images/su7.png" << endl;
        return;
    }

    // 将车辆模型旋转 90 度（逆时针）
    Mat rotated_su7;
    rotate(img_su7, rotated_su7, ROTATE_90_COUNTERCLOCKWISE);

    // 计算车辆模型的中心放置尺寸（根据拼接图尺寸调整）
    double target_width = totalWidth * 0.5; // 设置为总宽度的 50%
    double scale        = target_width / rotated_su7.cols;
    Size   target_size(target_width, rotated_su7.rows * scale);

    // 缩放车辆模型
    Mat resized_su7;
    resize(rotated_su7, resized_su7, target_size);

    // 分离通道，获取 Alpha 通道
    vector<Mat> channels;
    split(resized_su7, channels);

    // 创建三通道彩色图像和 Alpha 遮罩
    Mat color_su7;
    Mat alpha_mask;

    if (channels.size() == 4)
    { // BGRA 图像
        vector<Mat> color_channels = {channels[0], channels[1], channels[2]};
        merge(color_channels, color_su7);
        alpha_mask = channels[3]; // Alpha 通道
    }
    else
    {
        color_su7  = resized_su7;
        alpha_mask = Mat::ones(resized_su7.size(), CV_8UC1) * 255;
    }

    // 将 Alpha 遮罩转换为浮点型并进行归一化
    alpha_mask.convertTo(alpha_mask, CV_32F, 1.0 / 255.0);

    // 计算车辆模型在结果图像中的位置（居中放置）
    int x = (totalWidth - resized_su7.cols) / 2;
    int y = (totalHeight - resized_su7.rows) / 2;

    // 在结果图像中创建 ROI
    Mat roi_su7 = result(Rect(x, y, resized_su7.cols, resized_su7.rows));
    Mat roi_su7_float;
    roi_su7.convertTo(roi_su7_float, CV_32FC3);

    // 将车辆模型转换为浮点型
    Mat color_su7_float;
    color_su7.convertTo(color_su7_float, CV_32FC3);

    // 使用 Alpha 混合公式：结果 = alpha * 前景 + (1 - alpha) * 背景
    for (int i = 0; i < roi_su7.rows; i++)
    {
        for (int j = 0; j < roi_su7.cols; j++)
        {
            float alpha                   = alpha_mask.at<float>(i, j);
            roi_su7_float.at<Vec3f>(i, j) = alpha * color_su7_float.at<Vec3f>(i, j) +
                (1.0f - alpha) * roi_su7_float.at<Vec3f>(i, j);
        }
    }

    // 转换回 8 位无符号整数
    roi_su7_float.convertTo(roi_su7, CV_8UC3);

    // 保存最终结果
    imwrite("build/stitched_result_with_su7.jpg", result);
    cout << "[SUCCESS] Panoramic stitching completed, result saved: build/stitched_result_with_su7.jpg" << endl;
}

// ========================================
// 主函数
// ========================================

/**
 * @brief 环视系统主函数
 * @description 完成完整的 AVM 处理流水线：图像去畸变、角点检测、透视变换、图像拼接
 * @return 程序执行状态
 */
int main()
{
    cout << "===========================================" << endl;
    cout << "[SYSTEM] Around View Monitor (AVM) System Started" << endl;
    cout << "===========================================" << endl;

    // 创建输出目录
    struct stat st = {0};
    if (stat("build", &st) == -1)
    {
#ifdef _WIN32
        if (_mkdir("build") == 0)
        {
#else
        if (mkdir("build", 0755) == 0)
        {
#endif
            cout << "[INIT] Output directory created: build/" << endl;
        }
        else
        {
            cout << "[WARNING] Unable to create output directory, will use current directory" << endl;
        }
    }

    // 初始化角点容器
    g_corner_front = std::vector<cv::Point2f>(8);
    g_corner_back  = std::vector<cv::Point2f>(8);
    g_corner_left  = std::vector<cv::Point2f>(8);
    g_corner_right = std::vector<cv::Point2f>(8);

    // 设置相机参数
    float fish_scale   = 0.5f;
    float focal_length = 910.0f;
    int   dx           = 3;
    int   dy           = 3;
    int   fish_width   = 1280;
    int   fish_height  = 960;
    float undis_scale  = 1.55f;

    // 鱼眼到去畸变参数
    g_fish2undis_params = {-0.05611147, -0.05377447, 0.0115717, 0.0030788};

    // 构建去畸变内参矩阵
    g_intrinsic_undis = (cv::Mat_<float>(3, 3) << focal_length / dx * fish_scale, 0, fish_width / 2 * undis_scale, 0, focal_length / dy * fish_scale, fish_height / 2 * undis_scale, 0, 0, 1);

    // 构建原始内参矩阵
    g_intrinsic = (cv::Mat_<float>(3, 3) << focal_length / dx, 0, fish_width / 2, 0, focal_length / dy, fish_height / 2, 0, 0, 1);

    cout << "[INIT] Camera parameters initialization completed" << endl;

    // 读取四个方向的鱼眼图像
    cout << "[STEP 1] Reading fisheye images..." << endl;
    cv::Mat image_f = imread("assets/images/front.png");
    cv::Mat image_b = imread("assets/images/back.png");
    cv::Mat image_l = imread("assets/images/left.png");
    cv::Mat image_r = imread("assets/images/right.png");

    if (image_f.empty() || image_b.empty() || image_l.empty() || image_r.empty())
    {
        cout << "[ERROR] Unable to read input image files" << endl;
        return -1;
    }
    cout << "[SUCCESS] Successfully read 4 fisheye images" << endl;

    // 创建去畸变处理对象
    cout << "[STEP 2] Starting image undistortion processing..." << endl;
    Undistort            undistort_handle;
    std::vector<cv::Mat> undis2dis_front, undis2dis_back, undis2dis_left, undis2dis_right;

    // 执行图像去畸变
    cv::Mat front_undis = undistort_handle.undistort_func(image_f, undis2dis_front);
    cv::Mat back_undis  = undistort_handle.undistort_func(image_b, undis2dis_back);
    cv::Mat left_undis  = undistort_handle.undistort_func(image_l, undis2dis_left);
    cv::Mat right_undis = undistort_handle.undistort_func(image_r, undis2dis_right);

    // 保存去畸变图像
    cv::imwrite("build/front_undis.jpg", front_undis);
    cv::imwrite("build/back_undis.jpg", back_undis);
    cv::imwrite("build/left_undis.jpg", left_undis);
    cv::imwrite("build/right_undis.jpg", right_undis);
    cout << "[SUCCESS] Undistorted image processing completed and saved" << endl;

    // 检测标定板角点
    cout << "[STEP 3] Detecting calibration board corners..." << endl;
    bool success = true;
    success &= detectPoints(image_f, 20000, 0.5, g_corner_front, 0, ImageType::IMAGE_FRONT);
    success &= detectPoints(image_b, 20000, 0.5, g_corner_back, 0, ImageType::IMAGE_BACK);
    success &= detectPoints(image_l, 20000, 0.5, g_corner_left, 0, ImageType::IMAGE_LEFT);
    success &= detectPoints(image_r, 20000, 0.5, g_corner_right, 0, ImageType::IMAGE_RIGHT);

    if (!success)
    {
        cout << "[WARNING] Some corner detection failed, continuing processing..." << endl;
    }
    else
    {
        cout << "[SUCCESS] All directional corner detection completed" << endl;
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
    cv::imwrite("build/front_undis_1.jpg", front_undis);
    cv::imwrite("build/back_undis_1.jpg", back_undis);
    cv::imwrite("build/left_undis_1.jpg", left_undis);
    cv::imwrite("build/right_undis_1.jpg", right_undis);
    cout << "[SUCCESS] Corner-marked images saved" << endl;

    // 初始化鸟瞰图参数
    init_params();

    // 重新读取去畸变图像用于透视变换
    cv::Mat undisimage_f = imread("build/front_undis.jpg");
    cv::Mat undisimage_b = imread("build/back_undis.jpg");
    cv::Mat undisimage_l = imread("build/left_undis.jpg");
    cv::Mat undisimage_r = imread("build/right_undis.jpg");

    // 计算单应性矩阵
    cout << "[STEP 4] Calculating perspective transformation matrices..." << endl;
    g_Homo_F = cv::findHomography(g_corner_front, g_corner_bird_front, 0);
    g_Homo_B = cv::findHomography(g_corner_back, g_corner_bird_back, 0);
    g_Homo_L = cv::findHomography(g_corner_left, g_corner_bird_left, 0);
    g_Homo_R = cv::findHomography(g_corner_right, g_corner_bird_right, 0);
    cout << "[SUCCESS] Perspective transformation matrices calculation completed" << endl;

    // 执行透视变换生成鸟瞰图
    cout << "[STEP 5] Generating bird's eye view..." << endl;
    cv::Mat bird_front_image, bird_back_image, bird_left_image, bird_right_image;

    cv::warpPerspective(undisimage_f, bird_front_image, g_Homo_F, cv::Size(792, 305), cv::INTER_LINEAR);
    cv::warpPerspective(undisimage_b, bird_back_image, g_Homo_B, cv::Size(792, 305), cv::INTER_LINEAR);
    cv::warpPerspective(undisimage_l, bird_left_image, g_Homo_L, cv::Size(1131, 281), cv::INTER_LINEAR);
    cv::warpPerspective(undisimage_r, bird_right_image, g_Homo_R, cv::Size(1131, 281), cv::INTER_LINEAR);

    // 保存初始鸟瞰图图像
    cv::imwrite("build/bird_front.jpg", bird_front_image);
    cv::imwrite("build/bird_back.jpg", bird_back_image);
    cv::imwrite("build/bird_left.jpg", bird_left_image);
    cv::imwrite("build/bird_right.jpg", bird_right_image);
    cout << "[SUCCESS] Bird's eye view generation completed" << endl;

    // 旋转鸟瞰图以纠正朝向
    cout << "[STEP 6] Adjusting bird's eye view orientation..." << endl;
    rotate("build/bird_front.jpg", "build/bird_front_2.jpg", 0);
    rotate("build/bird_back.jpg", "build/bird_back_2.jpg", 180);
    rotate("build/bird_left.jpg", "build/bird_left_2.jpg", 90);
    rotate("build/bird_right.jpg", "build/bird_right_2.jpg", -90);
    cout << "[SUCCESS] Bird's eye view orientation adjustment completed" << endl;

    // 执行全景拼接
    cout << "[STEP 7] Performing panoramic image stitching..." << endl;
    join();

    cout << "===========================================" << endl;
    cout << "[SYSTEM] Around View Monitor system processing completed" << endl;
    cout << "===========================================" << endl;
    return 0;
}