/**
 * @file utils.cpp
 * @brief 系统通用辅助函数与实用工具实现
 */

#include "utils.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

std::string utf8_to_gbk(const std::string& str)
{
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    if (len <= 0) return "";
    std::wstring wstr(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], len);

    int len2 = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (len2 <= 0) return "";
    std::string gbk_str(len2, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &gbk_str[0], len2, NULL, NULL);

    // 移除转换后字符串尾部冗余的 '\0' 字符
    if (!gbk_str.empty() && gbk_str.back() == '\0')
    {
        gbk_str.pop_back();
    }
    return gbk_str;
}
#else
std::string utf8_to_gbk(const std::string& str)
{
    return str;
}
#endif
