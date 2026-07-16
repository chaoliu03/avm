/**
 * @file stitcher.cpp
 * @brief 图像拼接与融合模块实现
 */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "stitcher.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifdef _WIN32
/**
 * @brief 将 UTF-8 编码的 std::string 转换为系统本地多字节编码（Windows 下为 GBK/ANSI）
 * @description 解决 Windows 上 OpenCV 窗口标题中文乱码的问题
 */
static std::string utf8_to_gbk(const std::string& str)
{
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    if (len <= 0) return "";
    std::wstring wstr(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], len);

    int len2 = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (len2 <= 0) return "";
    std::string gbk_str(len2, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &gbk_str[0], len2, NULL, NULL);

    // 移除转换后字符串尾部冗余的 '\0' 字符，避免与 wstring 带来的长度不匹配
    if (!gbk_str.empty() && gbk_str.back() == '\0')
    {
        gbk_str.pop_back();
    }
    return gbk_str;
}
#else
static std::string utf8_to_gbk(const std::string& str)
{
    return str;
}
#endif

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
        cout << "[错误] 无法读取图像文件：" << src_image_path << endl;
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
    Mat img_front = imread(OUTPUT_DIR + "/bird_front_2.jpg");
    Mat img_back  = imread(OUTPUT_DIR + "/bird_back_2.jpg");
    Mat img_left  = imread(OUTPUT_DIR + "/bird_left_2.jpg");
    Mat img_right = imread(OUTPUT_DIR + "/bird_right_2.jpg");

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
        cout << "[错误] 无法加载一个或多个图像或遮罩文件" << endl;
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
        cout << "[错误] 无法加载车辆模型图像文件：assets/images/su7.png" << endl;
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
    imwrite(OUTPUT_DIR + "/stitched_result_with_su7.jpg", result);
    cout << "[成功] 全景图拼接完成，结果已保存：" << OUTPUT_DIR << "/stitched_result_with_su7.jpg" << endl;
}

// 图像交互缩放与拖拽状态结构体
struct PanZoomState
{
    cv::Mat     canvas;
    double      scale              = 1.0;
    cv::Point2d offset             = cv::Point2d(0, 0);
    bool        is_dragging        = false;
    cv::Point   drag_start_mouse;
    cv::Point2d drag_start_offset;
    std::string window_name;

    void update_display()
    {
        // 获取窗口当前的实际图像显示区域大小，从而进行自适应分辨率的 warp 映射
        cv::Rect rect = cv::getWindowImageRect(window_name);
        int win_w = rect.width > 0 ? rect.width : 1000;
        int win_h = rect.height > 0 ? rect.height : 1000;

        cv::Mat display_img;
        cv::Mat M = (cv::Mat_<double>(2, 3) << scale, 0, offset.x, 0, scale, offset.y);
        // 从原始高清大画布直接 warp 到窗口所占的分辨率大小，保证像素不被压缩，且缩放平移极致流畅
        cv::warpAffine(canvas, display_img, M, cv::Size(win_w, win_h), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(40, 40, 40));
        cv::imshow(window_name, display_img);
    }
};

// 鼠标交互回调函数
static void onMouse(int event, int x, int y, int flags, void* userdata)
{
    PanZoomState* state = static_cast<PanZoomState*>(userdata);
    if (!state)
        return;

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        state->is_dragging        = true;
        state->drag_start_mouse   = cv::Point(x, y);
        state->drag_start_offset = state->offset;
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        state->is_dragging = false;
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (state->is_dragging)
        {
            cv::Point   curr_mouse(x, y);
            cv::Point   diff = curr_mouse - state->drag_start_mouse;
            state->offset    = state->drag_start_offset + cv::Point2d(diff.x, diff.y);
            state->update_display();
        }
    }
    else if (event == cv::EVENT_MOUSEWHEEL)
    {
        // 获取滚轮缩放方向
        int    delta  = cv::getMouseWheelDelta(flags);
        double factor = (delta > 0) ? 1.1 : 0.9;

        // 限制缩放比例在 0.05 到 15.0 之间
        double new_scale = state->scale * factor;
        if (new_scale < 0.05)
            new_scale = 0.05;
        if (new_scale > 15.0)
            new_scale = 15.0;

        // 以当前鼠标指针所在的窗口位置为中心进行平滑缩放
        cv::Point2d mouse_pos(x, y);
        state->offset = mouse_pos - (new_scale / state->scale) * (mouse_pos - state->offset);
        state->scale  = new_scale;

        state->update_display();
    }
}

void showUndistortStitched(const cv::Mat& front, const cv::Mat& back, const cv::Mat& left, const cv::Mat& right)
{
    // 获取相机原始尺寸（不进行像素压缩，保留 100% 细节）
    int w = front.cols;
    int h = front.rows;

    // 旋转左右侧视图以纠正朝向，使得车头方向均朝上，车身均朝向中心：
    // 左侧视图顺时针旋转 90 度，尺寸变为 h x w
    cv::Mat l_rotated;
    cv::rotate(left, l_rotated, cv::ROTATE_90_COUNTERCLOCKWISE);

    // 右侧视图逆时针旋转 90 度，尺寸变为 h x w
    cv::Mat r_rotated;
    cv::rotate(right, r_rotated, cv::ROTATE_90_CLOCKWISE);

    // 计算高分辨率十字画布尺寸：
    // 总宽度 = 左侧宽(h) + 中间宽(w) + 右侧宽(h)
    // 总高度 = 前向高(h) + 中间高(w) + 后向高(h)
    int canvas_w = h + w + h;
    int canvas_h = h + w + h;
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(40, 40, 40));

    // 拼入前视图 (中上)
    front.copyTo(canvas(cv::Rect(h, 0, w, h)));

    // 拼入左视图 (左中)
    l_rotated.copyTo(canvas(cv::Rect(0, h, h, w)));

    // 拼入右视图 (右中)
    r_rotated.copyTo(canvas(cv::Rect(h + w, h, h, w)));

    // 拼入后视图 (中下)
    back.copyTo(canvas(cv::Rect(h, h + w, w, h)));

    // 实例化并初始化交互状态
    PanZoomState state;
    state.canvas      = canvas;
    state.window_name = utf8_to_gbk("去畸变方位拼接展示 (鼠标滚轮缩放，左键拖拽平移，按任意键继续)");

    // 设置初始缩放比例：将原始高分辨率画布等比缩放到 1000x1000 视口中显示
    state.scale       = 1000.0 / canvas_w;
    state.offset      = cv::Point2d(0, 0);

    // 创建可手动改变大小的窗口并预设窗口尺寸为 1000x1000
    cv::namedWindow(state.window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(state.window_name, 1000, 1000);
    cv::setMouseCallback(state.window_name, onMouse, &state);

    // 绘制初始图像
    state.update_display();

    // 等待按键关闭
    cv::waitKey(0);
    cv::destroyWindow(state.window_name);
}