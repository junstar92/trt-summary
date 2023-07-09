#pragma once
#include <NvInferRuntime.h>
#include <string>
#include <cassert>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
    using Severity = nvinfer1::ILogger::Severity;
public:
    explicit Logger(Severity severity = Severity::kINFO) : severity_(severity)
    {
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity_ >= severity)
            std::cout << severityPrefix(severity) << std::string(msg) << std::endl;
    }

    void setReportableSeverity(Severity severity) noexcept
    {
        severity_ = severity;
    }

private:
    Severity severity_;

    std::string severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }
};