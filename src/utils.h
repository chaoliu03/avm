/**
 * @file utils.h
 * @brief 系统通用辅助函数与实用工具声明
 */

#ifndef AVM_UTILS_H
#define AVM_UTILS_H

#include <string>

/**
 * @brief 将 UTF-8 编码的 std::string 转换为系统本地多字节编码（Windows 下为 GBK/ANSI）
 * @description 解决 Windows 上 OpenCV 窗口或文件路径中文乱码的问题
 * @param str 输入的 UTF-8 编码字符串
 * @return 转换后的本地 GBK/ANSI 编码字符串
 */
std::string utf8_to_gbk(const std::string& str);

#endif // AVM_UTILS_H
