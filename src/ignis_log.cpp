#include "ignis_log.h"

#include <windows.h>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cstdlib>

static FILE* g_logFile = nullptr;
static wchar_t g_logPath[MAX_PATH] = {0};

void SetLogPath(const char* path) {
    if (g_logFile) {
        fclose(g_logFile);
        g_logFile = nullptr;
    }
    if (path && path[0]) {
        MultiByteToWideChar(CP_UTF8, 0, path, -1, g_logPath, MAX_PATH);
    } else {
        g_logPath[0] = L'\0';
    }
}

static void EnsureLogFile() {
    if (g_logFile) return;

    wchar_t finalPath[MAX_PATH];

    if (g_logPath[0] != L'\0') {
        wcscpy_s(finalPath, g_logPath);
    } else {
        // Default: next to the host executable
        GetModuleFileNameW(NULL, finalPath, MAX_PATH);
        wchar_t* lastSlash = wcsrchr(finalPath, L'\\');
        if (lastSlash) *(lastSlash + 1) = L'\0';
        wcscat_s(finalPath, L"ignis-rt.log");
    }

    g_logFile = _wfopen(finalPath, L"a");
}

void Log(const wchar_t* fmt, ...) {
    wchar_t buf[2048];
    va_list args;
    va_start(args, fmt);
    vswprintf_s(buf, _countof(buf), fmt, args);
    va_end(args);

    OutputDebugStringW(buf);

    EnsureLogFile();
    if (g_logFile) {
        fwprintf(g_logFile, L"%s", buf);
        fflush(g_logFile);
    }
}
