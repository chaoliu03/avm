/**
 * @file avm.cpp
 * @brief Around View Monitor (AVM) System Implementation
 * @description This program implements a surround view system based on four fisheye cameras,
 *              including image undistortion, corner detection, perspective transformation, and image stitching
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <sys/stat.h> // For creating directories
#ifdef _WIN32
#include <direct.h> // For _mkdir on Windows
#endif

using namespace cv;
using namespace std;

// Constants definition
#define IMAGE_BACK_PIXEL_Y  643 // Y offset for back view image in stitched result
#define IMAGE_RIGHT_PIXEL_X 398 // X offset for right view image in stitched result

// Global variables
cv::Mat   g_intrinsic_undis;   // Undistortion intrinsic matrix
cv::Mat   g_intrinsic;         // Original intrinsic matrix
cv::Vec4d g_fish2undis_params; // Fisheye to undistortion transformation parameters

// Corner coordinates for four directions
std::vector<cv::Point2f> g_corner_front; // Front view corners
std::vector<cv::Point2f> g_corner_back;  // Back view corners
std::vector<cv::Point2f> g_corner_left;  // Left view corners
std::vector<cv::Point2f> g_corner_right; // Right view corners

/**
 * @brief Image type enumeration
 */
typedef enum
{
    IMAGE_FRONT = 0, // Front view image
    IMAGE_BACK,      // Back view image
    IMAGE_LEFT,      // Left view image
    IMAGE_RIGHT      // Right view image
} ImageType;

/**
 * @brief Image undistortion processing class
 * @description Responsible for converting fisheye images to undistorted images
 */
class Undistort
{
public:
    /**
     * @brief Constructor, initializes undistortion parameters
     */
    Undistort();

    /**
     * @brief Perform image undistortion
     * @param img Input fisheye image
     * @param remap_table Remapping table (output parameter)
     * @return Undistorted image
     */
    cv::Mat undistort_func(cv::Mat img, vector<cv::Mat>& remap_table);

private:
    /**
     * @brief Generate undistortion remapping table
     * @param remap_table Remapping table (output)
     * @param undist_w Undistorted image width
     * @param undist_h Undistorted image height
     * @param intrinsic_undis Undistortion intrinsic matrix
     * @param intrinsic_fish Fisheye intrinsic matrix
     * @param undis_param Undistortion parameters
     * @param fish_scale Fisheye scaling factor
     */
    void getUndistortMap(vector<cv::Mat>& remap_table, int undist_w, int undist_h, cv::Mat intrinsic_undis, cv::Mat intrinsic_fish, cv::Vec4d undis_param, float fish_scale);

    // Member variables
    int       m_undis_width;       // Undistorted image width
    int       m_undis_height;      // Undistorted image height
    float     m_fish_scale;        // Fisheye scaling factor
    float     m_focal_length;      // Focal length
    float     m_dx;                // X direction pixel spacing
    float     m_dy;                // Y direction pixel spacing
    float     m_fish_width;        // Fisheye image width
    float     m_fish_height;       // Fisheye image height
    float     m_undis_scale;       // Undistortion scaling factor
    cv::Vec4d m_undis2fish_params; // Undistortion to fisheye parameters
    cv::Mat   m_intrinsic_undis;   // Undistortion intrinsic matrix
    cv::Mat   m_intrinsic;         // Original intrinsic matrix
};

/**
 * @brief Undistort class constructor
 * @description Initialize various parameters and intrinsic matrices for fisheye camera
 */
Undistort::Undistort()
{
    // Initialize camera parameters
    m_fish_scale   = 0.5f;    // Fisheye scaling factor
    m_focal_length = 910.0f;  // Focal length
    m_dx           = 3.0f;    // X direction pixel spacing
    m_dy           = 3.0f;    // Y direction pixel spacing
    m_fish_width   = 1280.0f; // Fisheye image width
    m_fish_height  = 960.0f;  // Fisheye image height
    m_undis_scale  = 1.55f;   // Undistortion scaling factor

    // Polynomial parameters from undistortion to fisheye
    m_undis2fish_params = {0.18238692, -0.08579553, 0.03366532, -0.00561911};

    // Calculate undistorted image dimensions
    m_undis_width  = static_cast<int>(m_undis_scale * m_fish_width);
    m_undis_height = static_cast<int>(m_undis_scale * m_fish_height);

    // Build undistortion intrinsic matrix
    m_intrinsic_undis = (cv::Mat_<float>(3, 3) << m_focal_length / m_dx * m_fish_scale, 0, m_fish_width / 2 * m_undis_scale, 0, m_focal_length / m_dy * m_fish_scale, m_fish_height / 2 * m_undis_scale, 0, 0, 1);

    cout << "[INFO] Undistortion intrinsic matrix initialization completed" << endl;

    // Build original intrinsic matrix
    m_intrinsic = (cv::Mat_<float>(3, 3) << m_focal_length / m_dx, 0, m_fish_width / 2, 0, m_focal_length / m_dy, m_fish_height / 2, 0, 0, 1);

    cout << "[INFO] Original intrinsic matrix initialization completed" << endl;
}

/**
 * @brief OpenCV-style point coordinate transformation
 * @param warp_xy Output transformed coordinates
 * @param map_center_h Mapping center Y coordinate
 * @param map_center_w Mapping center X coordinate
 * @param x_ Input X coordinate
 * @param y_ Input Y coordinate
 * @param scale Scaling factor
 */
void warpPointOpencv(cv::Vec2f& warp_xy, float map_center_h, float map_center_w, float x_, float y_, float scale)
{
    warp_xy[0] = x_ * scale + map_center_w;
    warp_xy[1] = y_ * scale + map_center_h;
}

/**
 * @brief Transform from undistorted coordinate system to fisheye coordinate system
 * @description Use polynomial distortion model to implement coordinate transformation
 * @param fish_scale Fisheye scaling factor
 * @param f_dx X direction focal length to pixel spacing ratio
 * @param f_dy Y direction focal length to pixel spacing ratio
 * @param large_center_h Undistorted image center Y coordinate
 * @param large_center_w Undistorted image center X coordinate
 * @param fish_center_h Fisheye image center Y coordinate
 * @param fish_center_w Fisheye image center X coordinate
 * @param undis_param Undistortion parameters
 * @param x Input X coordinate
 * @param y Input Y coordinate
 * @return Transformed fisheye coordinates
 */
cv::Vec2f warpUndist2Fisheye(float fish_scale, float f_dx, float f_dy, float large_center_h, float large_center_w, float fish_center_h, float fish_center_w, const cv::Vec4d& undis_param, float x, float y)
{
    f_dx *= fish_scale;
    f_dy *= fish_scale;

    // Convert to normalized plane coordinates
    float y_ = (y - large_center_h) / f_dy;
    float x_ = (x - large_center_w) / f_dx;
    float r_ = static_cast<float>(sqrt(pow(x_, 2) + pow(y_, 2)));

    // Calculate angle
    float angle_undistorted = atan(r_);

    // Polynomial expansion
    float angle_undistorted_p2 = angle_undistorted * angle_undistorted;
    float angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted;
    float angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3;
    float angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5;
    float angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7;

    // Apply distortion model
    float angle_distorted = static_cast<float>(
        angle_undistorted + undis_param[0] * angle_undistorted_p3 +
        undis_param[1] * angle_undistorted_p5 +
        undis_param[2] * angle_undistorted_p7 +
        undis_param[3] * angle_undistorted_p9);

    // Calculate scaling ratio
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

    // Calculate remapping coordinates for each pixel
    for (int i = 0; i < undist_h; i++)
    {
        float* row_x = map_x.ptr<float>(i);
        float* row_y = map_y.ptr<float>(i);

        for (int j = 0; j < undist_w; j++)
        {
            cv::Vec2f xy = warpUndist2Fisheye(
                fish_scale, intrinsic_fish.at<float>(0, 0), intrinsic_fish.at<float>(1, 1), intrinsic_undis.at<float>(1, 2), intrinsic_undis.at<float>(0, 2), intrinsic_fish.at<float>(1, 2), intrinsic_fish.at<float>(0, 2), undis_param, static_cast<float>(j), static_cast<float>(i));

            // Boundary protection
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
    // Calibration initialization
    getUndistortMap(remap_table, m_undis_width, m_undis_height, m_intrinsic_undis, m_intrinsic, m_undis2fish_params, m_fish_scale);

    cv::Mat undis_img;
    cv::remap(img, undis_img, remap_table[0], remap_table[1], cv::INTER_LINEAR);
    return undis_img;
}

// ========================================
// Histogram processing related functions (based on OpenCV source code)
// ========================================

/**
 * @brief Calculate gradient of 256-level histogram
 * @description Used to detect peaks and valleys in histogram
 * @param piHist Input histogram
 * @param piHistGrad Output gradient histogram
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
 * @brief Smooth histogram using sliding window
 * @description Reduce noise in histogram, window size is 2*iWidth+1
 * @param piHist Input histogram
 * @param piHistSmooth Output smoothed histogram
 * @param iWidth Smoothing window radius
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
 * @brief Calculate grayscale histogram of image
 * @description Count the number of pixels for each grayscale level
 * @param img Input single-channel image
 * @param piHist Output 256-level histogram
 */
template <typename ArrayContainer>
static void icvGetIntensityHistogram256(const cv::Mat& img,
    ArrayContainer&                                    piHist)
{
    // Initialize histogram
    for (int i = 0; i < 256; i++)
        piHist[i] = 0;

    // Count occurrence of each pixel value
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
 * @brief Image binarization based on bimodal histogram
 * @description Automatically determine optimal threshold by analyzing image histogram to segment image into foreground and background
 * @param img Input single-channel image
 * @param fish_undis_flag Whether it is a fisheye undistortion flag
 * @return Binarized image
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

    // Allocate histogram buffers
    cv::AutoBuffer<int, 256> piHistIntensity(iNumBins);
    cv::AutoBuffer<int, 256> piHistSmooth(iNumBins);
    cv::AutoBuffer<int, 256> piHistGrad(iNumBins);
    cv::AutoBuffer<int>      piMaxPos(iMaxPos);

    // Calculate intensity histogram
    icvGetIntensityHistogram256(img, piHistIntensity);

    // Smooth histogram distribution
    icvSmoothHistogram256<1>(piHistIntensity, piHistSmooth);

    // Calculate gradient
    icvGradientOfHistogram256(piHistSmooth, piHistGrad);

    // Detect zero points (peaks)
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
        // No peaks detected, use median intensity
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
        // Single peak distribution
        iThresh = piMaxPos[0] / 2;
    }
    else if (iCntMaxima == 2)
    {
        // Bimodal distribution
        iThresh = (piMaxPos[0] + piMaxPos[1]) / 2;
    }
    else
    { // iCntMaxima >= 3, multimodal distribution
        // Check threshold for white part
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

        // Find zero point closest to white part
        for (unsigned n = 0; n < iCntMaxima - 1; ++n)
        {
            iIdxBGMax = n + 1;
            if (piMaxPos[n] < iIdxAccSum)
            {
                break;
            }
            iBrightMax = piMaxPos[n];
        }

        // Check threshold for black part
        int iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];

        // If too close to 255, skip to next peak
        if (piMaxPos[iIdxBGMax] >= 250 && iIdxBGMax + 1 < iCntMaxima)
        {
            iIdxBGMax++;
            iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];
        }

        // Find the largest black peak
        for (unsigned n = iIdxBGMax + 1; n < iCntMaxima; n++)
        {
            if (piHistIntensity[piMaxPos[n]] >= iMaxVal)
            {
                iMaxVal   = piHistIntensity[piMaxPos[n]];
                iIdxBGMax = n;
            }
        }

        // Set binarization threshold
        int iDist2 = (iBrightMax - piMaxPos[iIdxBGMax]) / 2;
        iThresh    = iBrightMax - iDist2;

        // Special processing for fisheye dark areas
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
// Image processing and corner detection related functions
// ========================================

/**
 * @brief Image contrast enhancement for corner detection
 * @description Enhance image contrast through gamma correction for subsequent corner detection
 * @param img Input three-channel image
 * @param contrast Contrast enhancement coefficient
 * @return Enhanced image
 */
cv::Mat imgAugForPointDetect(const cv::Mat img, float contrast)
{
    cv::Mat mat_float_tmp;

    // Convert to 32-bit floating-point three-channel image
    img.convertTo(mat_float_tmp, CV_32FC3);

    // Normalize to [0,1]
    mat_float_tmp = mat_float_tmp / 255.0f;

    // Find maximum value
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

    // Normalize again
    mat_float_tmp = mat_float_tmp / maxvalue_;

    // Apply gamma correction to enhance contrast
    pow(mat_float_tmp, contrast, mat_float_tmp);
    mat_float_tmp = mat_float_tmp * 255.0f;

    // Convert back to 8-bit unsigned integer
    cv::Mat contrast_img;
    mat_float_tmp.convertTo(contrast_img, CV_8UC3);

    return contrast_img;
}

/**
 * @brief Find rectangles (calibration board corners) in binary image
 * @description Find quadrilateral calibration boards through contour detection and polygon approximation
 * @param img Input binary image
 * @param valid_region_y Valid detection region in Y direction
 * @param max_sz Maximum area threshold
 * @param fish_scale Fisheye scaling factor
 * @param detect_points Detected corner points (output)
 * @param fish_undis_flag Fisheye undistortion flag
 */
void findRectangle(cv::Mat img, std::vector<float> valid_region_y, float max_sz, float fish_scale, std::vector<cv::Point2f>& detect_points, int fish_undis_flag)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;

    float min_y_thresh = valid_region_y[0];
    float max_y_thresh = valid_region_y[1];

    // Calculate minimum area threshold
    int min_size = static_cast<int>(max_sz * (pow(fish_scale, 2)));
    if (fish_undis_flag == 0)
    {
        auto temp = static_cast<float>(min_size) * 0.5f;
        min_size  = static_cast<int>(temp);
    }

    float approx_level = 10.0f * fish_scale;

    // Extract contours
    cv::findContours(img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect  contour_rect;
    cv::Point pt[4];

    for (int idx = (int)(contours.size() - 1); idx >= 0; --idx)
    {
        auto contour = contours[idx];
        contour_rect = boundingRect(contour);

        // Area filtering
        if (contour_rect.area() < min_size)
        {
            continue;
        }

        std::vector<cv::Point> approx_contour;
        // Polygon curve approximation
        cv::approxPolyDP(contour, approx_contour, approx_level, true);

        // Check if it's a quadrilateral
        if (approx_contour.size() == 4)
        {
            for (int i = 0; i < 4; ++i)
                pt[i] = approx_contour[i];

            // Check if center point is inside (black)
            int x_lable = (pt[0].x + pt[2].x) / 2;
            int y_lable = (pt[0].y + pt[2].y) / 2;
            if (img.at<uchar>(y_lable, x_lable) != 0)
            {
                continue;
            }

            // Calculate perimeter and area
            double p    = cv::arcLength(approx_contour, true);
            double area = cv::contourArea(approx_contour, false);

            // Calculate diagonal lengths
            double d1 = sqrt(cv::normL2Sqr<double>(pt[0] - pt[2]));
            double d2 = sqrt(cv::normL2Sqr<double>(pt[1] - pt[3]));

            // Calculate side lengths
            double d3 = sqrt(cv::normL2Sqr<double>(pt[0] - pt[1]));
            double d4 = sqrt(cv::normL2Sqr<double>(pt[1] - pt[2]));

            // Shape validation: check if it's close to square
            if (!(d3 * 5 > d4 && d4 * 5 > d3 && d3 * d4 < area * 10 &&
                    area > min_size && d1 >= 0.1 * p && d2 >= 0.1 * p))
            {
                continue;
            }

            // Y coordinate region validation
            float contour_y_average = static_cast<float>(
                                          approx_contour[0].y + approx_contour[1].y +
                                          approx_contour[2].y + approx_contour[3].y) /
                4.0f;

            if (contour_y_average < min_y_thresh || contour_y_average > max_y_thresh)
            {
                continue;
            }

            // Add detected corner points
            detect_points.push_back(approx_contour[0]);
            detect_points.push_back(approx_contour[1]);
            detect_points.push_back(approx_contour[2]);
            detect_points.push_back(approx_contour[3]);
        }
    }
}

// ========================================
// Sorting comparison functions
// ========================================

/**
 * @brief Comparison function for sorting contours by area in descending order
 */
bool cmp(std::vector<cv::Point> A, std::vector<cv::Point> B)
{
    return (contourArea(A) > contourArea(B));
}

/**
 * @brief Comparison function for sorting by Y coordinate in descending order
 */
bool cmpYmax(cv::Point A, cv::Point B)
{
    return (A.y > B.y);
}

/**
 * @brief Comparison function for sorting by X coordinate in descending order
 */
bool cmpXmax(cv::Point A, cv::Point B)
{
    return (A.x > B.x);
}

/**
 * @brief Comparison function for sorting by Y coordinate in ascending order
 */
bool cmpYmin(cv::Point A, cv::Point B)
{
    return (A.y < B.y);
}

/**
 * @brief Comparison function for sorting by X coordinate in ascending order
 */
bool cmpXmin(cv::Point A, cv::Point B)
{
    return (A.x < B.x);
}

/**
 * @brief Sort detected corner points
 * @description Sort 8 corner points in order from top to bottom, left to right
 * @param points Input 8 corner points
 * @return Sorted sequence of corner points
 */
static std::vector<cv::Point2f> detectPointsSort(std::vector<cv::Point2f> points)
{
    // Sort by Y coordinate from small to large
    sort(points.begin(), points.end(), cmpYmin);

    // Divide into two rows, 4 points per row
    std::vector<cv::Point2f> mid1(points.begin(), points.begin() + 4);
    std::vector<cv::Point2f> mid2(points.begin() + 4, points.begin() + 8);

    // Sort by X coordinate from small to large within each row
    sort(mid1.begin(), mid1.end(), cmpXmin);
    sort(mid2.begin(), mid2.end(), cmpXmin);

    // Merge results
    std::vector<cv::Point2f> sortedPoints(mid1.begin(), mid1.end());
    sortedPoints.insert(sortedPoints.end(), mid2.begin(), mid2.end());

    return sortedPoints;
}

// ========================================
// Coordinate transformation related functions
// ========================================

/**
 * @brief Inverse point coordinate transformation
 * @param warp_xy Output transformed coordinates
 * @param map_center_h Mapping center Y coordinate
 * @param map_center_w Mapping center X coordinate
 * @param x_ Input X coordinate
 * @param y_ Input Y coordinate
 * @param scale Scaling factor
 */
void warpPointInverse(cv::Vec2f& warp_xy, float map_center_h, float map_center_w, float x_, float y_, float scale)
{
    warp_xy[0] = x_ * scale + map_center_w;
    warp_xy[1] = y_ * scale + map_center_h;
}

/**
 * @brief Transform from fisheye coordinate system to undistorted coordinate system
 * @description Implement coordinate transformation from fisheye image to normal perspective image
 * @param fish_scale Fisheye scaling factor
 * @param f_dx X direction focal length to pixel spacing ratio
 * @param f_dy Y direction focal length to pixel spacing ratio
 * @param undis_center_h Undistorted image center Y coordinate
 * @param undis_center_w Undistorted image center X coordinate
 * @param fish_center_h Fisheye image center Y coordinate
 * @param fish_center_w Fisheye image center X coordinate
 * @param undis_param Undistortion parameters
 * @param x Input X coordinate
 * @param y Input Y coordinate
 * @return Transformed undistorted coordinates
 */
cv::Vec2f warpFisheye2Undist(float fish_scale, float f_dx, float f_dy, float undis_center_h, float undis_center_w, float fish_center_h, float fish_center_w, cv::Vec4d undis_param, float x, float y)
{
    // Pixel projection to normalized imaging coordinate system
    float y_          = (y - fish_center_h) / f_dy;
    float x_          = (x - fish_center_w) / f_dx;
    float r_distorted = static_cast<float>(sqrt(pow(x_, 2) + pow(y_, 2)));

    // Calculate refraction angle according to distortion formula
    float r_distorted_p2 = r_distorted * r_distorted;
    float r_distorted_p3 = r_distorted_p2 * r_distorted;
    float r_distorted_p4 = r_distorted_p2 * r_distorted_p2;
    float r_distorted_p5 = r_distorted_p2 * r_distorted_p3;

    float angle_undistorted = static_cast<float>(
        r_distorted + undis_param[0] * r_distorted_p2 +
        undis_param[1] * r_distorted_p3 + undis_param[2] * r_distorted_p4 +
        undis_param[3] * r_distorted_p5);

    // Calculate scale factor in normalized image
    float r_undistorted = tanf(angle_undistorted);
    float scale         = r_undistorted / (r_distorted + 0.00001f);

    cv::Vec2f warp_xy;

    // Convert to real image coordinate system
    float xx = (x - fish_center_w) * fish_scale;
    float yy = (y - fish_center_h) * fish_scale;

    // Project to undistorted image coordinate system
    warpPointInverse(warp_xy, undis_center_h, undis_center_w, xx, yy, scale);

    return warp_xy;
}

/**
 * @brief Detect calibration board corners
 * @description Detect 2x4 arranged calibration board corners in image and convert them to undistorted coordinate system
 * @param img Input image
 * @param max_sz Maximum area threshold
 * @param fish_scale Fisheye scaling factor
 * @param detect_points Detected corner points (output)
 * @param fish_undis_flag Fisheye undistortion flag
 * @param src_image_type Source image type
 * @return Whether 8 corner points were successfully detected
 */
bool detectPoints(cv::Mat img, float max_sz, float fish_scale, std::vector<cv::Point2f>& detect_points, int fish_undis_flag, ImageType src_image_type)
{
    // Set valid detection region (Y direction)
    float              max_y_thresh = static_cast<float>(0.7f * img.rows);
    float              min_y_thresh = static_cast<float>(0.2f * img.rows);
    std::vector<float> y_valid_area{min_y_thresh, max_y_thresh};

    // Convert to grayscale image
    if (img.channels() != 1)
    {
        cv::cvtColor(img, img, COLOR_BGR2GRAY);
    }

    // Image contrast enhancement
    float   contrast     = 2.5f;
    cv::Mat img_contrast = imgAugForPointDetect(img, contrast);

    // Save enhanced image for debugging
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

    // Histogram-based binarization
    cv::Mat img_thresh = icvBinarizationHistogramBased(img_contrast, 0);

    // Find rectangular corner points
    findRectangle(img_thresh, y_valid_area, max_sz, fish_scale, detect_points, fish_undis_flag);

    // If 8 corner points are not found, use backup solution
    if (detect_points.size() != 8)
    {
        // Adaptive threshold binarization
        cv::adaptiveThreshold(img_contrast, img_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 401, 5);
        detect_points.clear();
        findRectangle(img_thresh, y_valid_area, max_sz, fish_scale, detect_points, fish_undis_flag);
    }

    // Draw detected corner points on binary image for debugging
    for (int i = 0; i < detect_points.size(); i++)
    {
        cv::circle(img_thresh, detect_points[i], 1, cv::Scalar(0, 255, 0), 5);
    }

    // Save binarization results
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

    // Check if 8 corner points were successfully detected
    if (detect_points.size() != 8)
    {
        return false;
    }

    // Sub-pixel level corner optimization
    cv::cornerSubPix(img, detect_points, cv::Size(9, 9), cv::Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

    // Sort corner points
    detect_points = detectPointsSort(detect_points);

    // Get camera intrinsic parameters
    float     f_dx              = g_intrinsic.at<float>(0, 0);
    float     f_dy              = g_intrinsic.at<float>(1, 1);
    float     fish_center_x     = g_intrinsic.at<float>(0, 2);
    float     fish_center_y     = g_intrinsic.at<float>(1, 2);
    float     undis_center_x    = g_intrinsic_undis.at<float>(0, 2);
    float     undis_center_y    = g_intrinsic_undis.at<float>(1, 2);
    cv::Vec4d fish2undis_params = g_fish2undis_params;

    // Convert corner points from fisheye coordinate system to undistorted coordinate system
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
// Bird's eye view transformation related global variables and functions
// ========================================

// Target corner coordinates for four directions in bird's eye view
std::vector<cv::Point2i> g_corner_bird_front; // Front view bird's eye corner points
std::vector<cv::Point2i> g_corner_bird_back;  // Back view bird's eye corner points
std::vector<cv::Point2i> g_corner_bird_left;  // Left view bird's eye corner points
std::vector<cv::Point2i> g_corner_bird_right; // Right view bird's eye corner points

// Homography matrices for four directions
cv::Mat g_Homo_F; // Front view homography matrix
cv::Mat g_Homo_B; // Back view homography matrix
cv::Mat g_Homo_L; // Left view homography matrix
cv::Mat g_Homo_R; // Right view homography matrix

/**
 * @brief Initialize front view bird's eye target corner points
 * @description Set standard corner point positions for front view image in bird's eye view
 */
void init_des_front_points(void)
{
    g_corner_bird_front = std::vector<cv::Point2i>(8);

    // First row corner point X coordinates
    g_corner_bird_front[0].x = 136;
    g_corner_bird_front[1].x = 256;
    g_corner_bird_front[2].x = 536;
    g_corner_bird_front[3].x = 656;

    // Second row corner point X coordinates
    g_corner_bird_front[4].x = 136;
    g_corner_bird_front[5].x = 256;
    g_corner_bird_front[6].x = 536;
    g_corner_bird_front[7].x = 656;

    // First row corner point Y coordinates
    g_corner_bird_front[0].y = 85;
    g_corner_bird_front[1].y = 85;
    g_corner_bird_front[2].y = 85;
    g_corner_bird_front[3].y = 85;

    // Second row corner point Y coordinates
    g_corner_bird_front[4].y = 205;
    g_corner_bird_front[5].y = 205;
    g_corner_bird_front[6].y = 205;
    g_corner_bird_front[7].y = 205;
}

/**
 * @brief Initialize back view bird's eye target corner points
 * @description Set standard corner point positions for back view image in bird's eye view
 */
void init_des_back_points(void)
{
    g_corner_bird_back = std::vector<cv::Point2i>(8);

    // Corner point X coordinate settings
    g_corner_bird_back[0].x = 136;
    g_corner_bird_back[1].x = 256;
    g_corner_bird_back[2].x = 536;
    g_corner_bird_back[3].x = 656;
    g_corner_bird_back[4].x = 136;
    g_corner_bird_back[5].x = 256;
    g_corner_bird_back[6].x = 536;
    g_corner_bird_back[7].x = 656;

    // Corner point Y coordinate settings
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
 * @brief Initialize left view bird's eye target corner points
 * @description Set standard corner point positions for left view image in bird's eye view
 */
void init_des_left_points(void)
{
    g_corner_bird_left = std::vector<cv::Point2i>(8);

    // Corner point X coordinate settings
    g_corner_bird_left[0].x = 85;
    g_corner_bird_left[1].x = 205;
    g_corner_bird_left[2].x = 926;
    g_corner_bird_left[3].x = 1046;
    g_corner_bird_left[4].x = 85;
    g_corner_bird_left[5].x = 205;
    g_corner_bird_left[6].x = 926;
    g_corner_bird_left[7].x = 1046;

    // Corner point Y coordinate settings
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
 * @brief Initialize right view bird's eye target corner points
 * @description Set standard corner point positions for right view image in bird's eye view
 */
void init_des_right_points(void)
{
    g_corner_bird_right = std::vector<cv::Point2i>(8);

    // Corner point X coordinate settings
    g_corner_bird_right[0].x = 85;
    g_corner_bird_right[1].x = 205;
    g_corner_bird_right[2].x = 926;
    g_corner_bird_right[3].x = 1046;
    g_corner_bird_right[4].x = 85;
    g_corner_bird_right[5].x = 205;
    g_corner_bird_right[6].x = 926;
    g_corner_bird_right[7].x = 1046;

    // Corner point Y coordinate settings
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
 * @brief Initialize bird's eye target corner points for all directions
 * @description Call corner point initialization functions for each direction
 */
void init_des_points(void)
{
    init_des_front_points();
    init_des_back_points();
    init_des_left_points();
    init_des_right_points();
}

/**
 * @brief Initialize Around View Monitor system parameters
 * @description Initialize all necessary parameters and data structures
 */
void init_params(void)
{
    init_des_points();
}

// ========================================
// Image rotation and stitching related functions
// ========================================

/**
 * @brief Rotate image
 * @description Rotate image by specified angle and save to target path
 * @param src_image_path Source image path
 * @param dst_image_path Target image path
 * @param angle1 Rotation angle (degrees)
 */
void rotate(string src_image_path, string dst_image_path, double angle1)
{ // Read image
    Mat image = imread(src_image_path);
    if (image.empty())
    {
        cout << "[ERROR] Unable to read image file: " << src_image_path << endl;
        return;
    }

    // Get image dimensions
    int height = image.rows;
    int width  = image.cols;

    // Set rotation center as image center
    Point2f center(width / 2.0f, height / 2.0f);

    // Set rotation angle and scaling factor
    double angle = angle1;
    double scale = 1.0;

    // Get rotation matrix
    Mat rotationMatrix = getRotationMatrix2D(center, angle, scale);

    // Calculate rotated image dimensions to avoid content being cropped
    Rect bbox = RotatedRect(center, image.size(), angle).boundingRect();

    // Adjust translation part of rotation matrix to fit new image dimensions
    rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // Create output image
    Mat rotatedImage;

    // Apply affine transformation (rotation)
    warpAffine(image, rotatedImage, rotationMatrix, bbox.size());

    // Save rotated image
    imwrite(dst_image_path, rotatedImage);
}

/**
 * @brief Image stitching and merging
 * @description Merge source image into target image at specified position
 * @param src_image Source image
 * @param des_image Target image
 * @param src_image_type Source image type
 * @return Merged image
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
        // Process front and back view images
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
        // Process left and right view images
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
 * @brief Image stitching with mask
 * @description Use mask image for more precise image stitching
 * @param src_image Source image
 * @param mask_image Mask image
 * @param des_image Target image
 * @param src_image_type Source image type
 * @return Merged image
 */
cv::Mat ImageMergeWithMask(cv::Mat& src_image, cv::Mat& mask_image, cv::Mat& des_image, ImageType src_image_type)
{
    cv::Mat output(des_image.rows, des_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    return output;
}

/**
 * @brief Panoramic image stitching main function
 * @description Stitch four directional bird's eye view images into a complete panoramic view and add vehicle model
 */
void join()
{
    // Read four bird's eye view images and corresponding mask images
    Mat img_front = imread("build/bird_front_2.jpg");
    Mat img_back  = imread("build/bird_back_2.jpg");
    Mat img_left  = imread("build/bird_left_2.jpg");
    Mat img_right = imread("build/bird_right_2.jpg");

    // Resize images to fit stitching
    cv::resize(img_front, img_front, cv::Size(616, 237));
    cv::resize(img_back, img_back, cv::Size(616, 237));
    cv::resize(img_left, img_left, cv::Size(218, 880));
    cv::resize(img_right, img_right, cv::Size(218, 880));

    // Read mask images (three channels)
    Mat mask_front = imread("assets/masks/maskFront.jpg");
    Mat mask_back  = imread("assets/masks/maskBack.jpg");
    Mat mask_left  = imread("assets/masks/maskLeft.jpg");
    Mat mask_right = imread("assets/masks/maskRight.jpg"); // Check if images are loaded successfully
    if (img_front.empty() || img_back.empty() || img_left.empty() || img_right.empty() ||
        mask_front.empty() || mask_back.empty() || mask_left.empty() || mask_right.empty())
    {
        cout << "[ERROR] Unable to load one or more image or mask files" << endl;
        return;
    }

    // Normalize mask images to [0,1] range
    mask_front.convertTo(mask_front, CV_32FC3, 1.0 / 255.0);
    mask_back.convertTo(mask_back, CV_32FC3, 1.0 / 255.0);
    mask_left.convertTo(mask_left, CV_32FC3, 1.0 / 255.0);
    mask_right.convertTo(mask_right, CV_32FC3, 1.0 / 255.0);

    // Convert original images to floating point for multiplication
    Mat img_front_float, img_back_float, img_left_float, img_right_float;
    img_front.convertTo(img_front_float, CV_32FC3);
    img_back.convertTo(img_back_float, CV_32FC3);
    img_left.convertTo(img_left_float, CV_32FC3);
    img_right.convertTo(img_right_float, CV_32FC3);

    // Apply masks (pixel-wise multiplication)
    Mat masked_front, masked_back, masked_left, masked_right;
    multiply(img_front_float, mask_front, masked_front);
    multiply(img_back_float, mask_back, masked_back);
    multiply(img_left_float, mask_left, masked_left);
    multiply(img_right_float, mask_right, masked_right);

    // Convert back to 8-bit unsigned integer
    masked_front.convertTo(masked_front, CV_8UC3);
    masked_back.convertTo(masked_back, CV_8UC3);
    masked_left.convertTo(masked_left, CV_8UC3);
    masked_right.convertTo(masked_right, CV_8UC3);

    // Get image dimensions
    Size frontSize = masked_front.size();
    Size backSize  = masked_back.size();
    Size leftSize  = masked_left.size();
    Size rightSize = masked_right.size();

    // Calculate total dimensions of stitched image
    int totalWidth  = frontSize.width;
    int totalHeight = leftSize.height;

    // Create result image
    Mat result = Mat::zeros(totalHeight, totalWidth, CV_8UC3);

    // Stitch images - left side image
    masked_left.copyTo(result(Rect(0, 0, leftSize.width, leftSize.height)));

    // Stitch images - right side image (add to overlapping area with left side)
    Mat roi_right = result(Rect(totalWidth - rightSize.width, 0, rightSize.width, rightSize.height));
    add(roi_right, masked_right, roi_right);

    // Stitch images - front side image (add to existing content)
    Mat roi_front = result(Rect(0, 0, frontSize.width, frontSize.height));
    add(roi_front, masked_front, roi_front);

    // Stitch images - back side image (add to existing content)
    Mat roi_back = result(Rect(0, leftSize.height - backSize.height, backSize.width, backSize.height));
    add(roi_back, masked_back, roi_back);

    // Add vehicle model image
    Mat img_su7 = imread("assets/images/su7.png", IMREAD_UNCHANGED);
    if (img_su7.empty())
    {
        cout << "[ERROR] Unable to load vehicle model image file: assets/images/su7.png" << endl;
        return;
    }

    // Rotate vehicle model 90 degrees (counterclockwise)
    Mat rotated_su7;
    rotate(img_su7, rotated_su7, ROTATE_90_COUNTERCLOCKWISE);

    // Calculate target size of vehicle model (adjust according to stitched image size)
    double target_width = totalWidth * 0.5; // Set to 50% of total width
    double scale        = target_width / rotated_su7.cols;
    Size   target_size(target_width, rotated_su7.rows * scale);

    // Resize vehicle model
    Mat resized_su7;
    resize(rotated_su7, resized_su7, target_size);

    // Separate channels, get Alpha channel
    vector<Mat> channels;
    split(resized_su7, channels);

    // Create three-channel color image and Alpha mask
    Mat color_su7;
    Mat alpha_mask;

    if (channels.size() == 4)
    { // BGRA image
        vector<Mat> color_channels = {channels[0], channels[1], channels[2]};
        merge(color_channels, color_su7);
        alpha_mask = channels[3]; // Alpha channel
    }
    else
    {
        color_su7  = resized_su7;
        alpha_mask = Mat::ones(resized_su7.size(), CV_8UC1) * 255;
    }

    // Convert Alpha mask to floating point and normalize
    alpha_mask.convertTo(alpha_mask, CV_32F, 1.0 / 255.0);

    // Calculate position of vehicle model in result image (center placement)
    int x = (totalWidth - resized_su7.cols) / 2;
    int y = (totalHeight - resized_su7.rows) / 2;

    // Create ROI in result image
    Mat roi_su7 = result(Rect(x, y, resized_su7.cols, resized_su7.rows));
    Mat roi_su7_float;
    roi_su7.convertTo(roi_su7_float, CV_32FC3);

    // Convert vehicle model to floating point
    Mat color_su7_float;
    color_su7.convertTo(color_su7_float, CV_32FC3);

    // Use Alpha blending formula: result = alpha * foreground + (1 - alpha) * background
    for (int i = 0; i < roi_su7.rows; i++)
    {
        for (int j = 0; j < roi_su7.cols; j++)
        {
            float alpha                   = alpha_mask.at<float>(i, j);
            roi_su7_float.at<Vec3f>(i, j) = alpha * color_su7_float.at<Vec3f>(i, j) +
                (1.0f - alpha) * roi_su7_float.at<Vec3f>(i, j);
        }
    }

    // Convert back to 8-bit unsigned integer
    roi_su7_float.convertTo(roi_su7, CV_8UC3);

    // Save final result
    imwrite("build/stitched_result_with_su7.jpg", result);
    cout << "[SUCCESS] Panoramic stitching completed, result saved: build/stitched_result_with_su7.jpg" << endl;
}

// ========================================
// Main function
// ========================================

/**
 * @brief Around View Monitor system main function
 * @description Complete AVM processing pipeline: image undistortion, corner detection, perspective transformation, image stitching
 * @return Program execution status
 */
int main()
{
    cout << "===========================================" << endl;
    cout << "[SYSTEM] Around View Monitor (AVM) System Started" << endl;
    cout << "===========================================" << endl;

    // Create output directory
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

    // Initialize corner containers
    g_corner_front = std::vector<cv::Point2f>(8);
    g_corner_back  = std::vector<cv::Point2f>(8);
    g_corner_left  = std::vector<cv::Point2f>(8);
    g_corner_right = std::vector<cv::Point2f>(8);

    // Set camera parameters
    float fish_scale   = 0.5f;
    float focal_length = 910.0f;
    int   dx           = 3;
    int   dy           = 3;
    int   fish_width   = 1280;
    int   fish_height  = 960;
    float undis_scale  = 1.55f;

    // Fisheye to undistortion parameters
    g_fish2undis_params = {-0.05611147, -0.05377447, 0.0115717, 0.0030788};

    // Build undistortion intrinsic matrix
    g_intrinsic_undis = (cv::Mat_<float>(3, 3) << focal_length / dx * fish_scale, 0, fish_width / 2 * undis_scale, 0, focal_length / dy * fish_scale, fish_height / 2 * undis_scale, 0, 0, 1);

    // Build original intrinsic matrix
    g_intrinsic = (cv::Mat_<float>(3, 3) << focal_length / dx, 0, fish_width / 2, 0, focal_length / dy, fish_height / 2, 0, 0, 1);

    cout << "[INIT] Camera parameters initialization completed" << endl;

    // Read fisheye images from four directions
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

    // Create undistortion processing object
    cout << "[STEP 2] Starting image undistortion processing..." << endl;
    Undistort            undistort_handle;
    std::vector<cv::Mat> undis2dis_front, undis2dis_back, undis2dis_left, undis2dis_right;

    // Perform image undistortion
    cv::Mat front_undis = undistort_handle.undistort_func(image_f, undis2dis_front);
    cv::Mat back_undis  = undistort_handle.undistort_func(image_b, undis2dis_back);
    cv::Mat left_undis  = undistort_handle.undistort_func(image_l, undis2dis_left);
    cv::Mat right_undis = undistort_handle.undistort_func(image_r, undis2dis_right);

    // Save undistorted images
    cv::imwrite("build/front_undis.jpg", front_undis);
    cv::imwrite("build/back_undis.jpg", back_undis);
    cv::imwrite("build/left_undis.jpg", left_undis);
    cv::imwrite("build/right_undis.jpg", right_undis);
    cout << "[SUCCESS] Undistorted image processing completed and saved" << endl;

    // Detect calibration board corners
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

    // Mark detected corners on undistorted images
    for (int j = 0; j < 8; j++)
    {
        circle(front_undis, g_corner_front[j], 3, cv::Scalar(0, 255, 0), 1);
        circle(back_undis, g_corner_back[j], 3, cv::Scalar(0, 255, 0), 1);
        circle(left_undis, g_corner_left[j], 3, cv::Scalar(0, 255, 0), 1);
        circle(right_undis, g_corner_right[j], 3, cv::Scalar(0, 255, 0), 1);
    }

    // Save corner-marked images
    cv::imwrite("build/front_undis_1.jpg", front_undis);
    cv::imwrite("build/back_undis_1.jpg", back_undis);
    cv::imwrite("build/left_undis_1.jpg", left_undis);
    cv::imwrite("build/right_undis_1.jpg", right_undis);
    cout << "[SUCCESS] Corner-marked images saved" << endl;

    // Initialize bird's eye view parameters
    init_params();

    // Re-read undistorted images for perspective transformation
    cv::Mat undisimage_f = imread("build/front_undis.jpg");
    cv::Mat undisimage_b = imread("build/back_undis.jpg");
    cv::Mat undisimage_l = imread("build/left_undis.jpg");
    cv::Mat undisimage_r = imread("build/right_undis.jpg");

    // Calculate homography matrices
    cout << "[STEP 4] Calculating perspective transformation matrices..." << endl;
    g_Homo_F = cv::findHomography(g_corner_front, g_corner_bird_front, 0);
    g_Homo_B = cv::findHomography(g_corner_back, g_corner_bird_back, 0);
    g_Homo_L = cv::findHomography(g_corner_left, g_corner_bird_left, 0);
    g_Homo_R = cv::findHomography(g_corner_right, g_corner_bird_right, 0);
    cout << "[SUCCESS] Perspective transformation matrices calculation completed" << endl;

    // Perform perspective transformation to generate bird's eye view
    cout << "[STEP 5] Generating bird's eye view..." << endl;
    cv::Mat bird_front_image, bird_back_image, bird_left_image, bird_right_image;

    cv::warpPerspective(undisimage_f, bird_front_image, g_Homo_F, cv::Size(792, 305), cv::INTER_LINEAR);
    cv::warpPerspective(undisimage_b, bird_back_image, g_Homo_B, cv::Size(792, 305), cv::INTER_LINEAR);
    cv::warpPerspective(undisimage_l, bird_left_image, g_Homo_L, cv::Size(1131, 281), cv::INTER_LINEAR);
    cv::warpPerspective(undisimage_r, bird_right_image, g_Homo_R, cv::Size(1131, 281), cv::INTER_LINEAR);

    // Save initial bird's eye view images
    cv::imwrite("build/bird_front.jpg", bird_front_image);
    cv::imwrite("build/bird_back.jpg", bird_back_image);
    cv::imwrite("build/bird_left.jpg", bird_left_image);
    cv::imwrite("build/bird_right.jpg", bird_right_image);
    cout << "[SUCCESS] Bird's eye view generation completed" << endl;

    // Rotate bird's eye view images to correct orientation
    cout << "[STEP 6] Adjusting bird's eye view orientation..." << endl;
    rotate("build/bird_front.jpg", "build/bird_front_2.jpg", 0);
    rotate("build/bird_back.jpg", "build/bird_back_2.jpg", 180);
    rotate("build/bird_left.jpg", "build/bird_left_2.jpg", 90);
    rotate("build/bird_right.jpg", "build/bird_right_2.jpg", -90);
    cout << "[SUCCESS] Bird's eye view orientation adjustment completed" << endl;

    // Perform panoramic stitching
    cout << "[STEP 7] Performing panoramic image stitching..." << endl;
    join();

    cout << "===========================================" << endl;
    cout << "[SYSTEM] Around View Monitor system processing completed" << endl;
    cout << "===========================================" << endl;
    return 0;
}