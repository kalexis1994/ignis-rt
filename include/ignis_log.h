#pragma once

#include <cstdarg>

// Set custom log file path (call before any Log). Pass nullptr/empty to use default.
void SetLogPath(const char* path);

// Log function — writes to ignis-rt.log and OutputDebugStringW
void Log(const wchar_t* fmt, ...);

// Resolve a relative path using ignis_set_base_path root
#include <string>
std::string IgnisResolvePath(const char* relativePath);
