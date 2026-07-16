/**
 * @file stitcher.cpp
 * @brief 图像拼接与融合模块实现
 */

#include "stitcher.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ========================================
// 鸟瞰目标坐标初始化实现
// ========================================

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

void init_des_points(void)
{
    init_des_front_points();
    init_des_back_points();
    init_des_left_points();
    init_des_right_points();
}

void init_params(void)
{
    init_des_points();
}

// ========================================
// 旋转与拼接核心函数实现
// ========================================

/**
 * @brief 旋转图像（路径版）
 * @description 将图像旋转指定角度并保存到目标路径
 * @param src_image_path 源图像路径
 * @param dst_image_path 目标图像路径
 * @param angle1 旋转角度（度）
 */
void rotate(string src_image_path, string dst_image_path, double angle1)
{
    // 读取图像
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
    cv::rotate(img_su7, rotated_su7, ROTATE_90_COUNTERCLOCKWISE);

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
