#include <memory>
#include <fstream>
#include <logger.h>
#include <inspector.h>

Logger gLogger{nvinfer1::ILogger::Severity::kINFO};

void printHelp()
{
    std::cout << "usage: net-inspector (--onnx ONNX_FILE | --engine PLAN_FILE) [EXPORT_FILE]\n";
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 4) {
        printHelp();
        return EXIT_SUCCESS;
    }

    std::string option{argv[1]};
    std::string file_path{argv[2]};
    std::string export_file_path("");
    if (argc == 4) export_file_path.assign(argv[3]);
    Format format{};

    if (option.compare("--onnx") == 0) {
        format = Format::kONNX;
    }
    else if (option.compare("--engine") == 0) {
        format = Format::kPLAN;
    }
    else {
        printHelp();
        std::cout << "unknown option: " << option << std::endl;
    }

    bool is_export = false;
    std::ofstream ofs;
    if (export_file_path.length()) {
        ofs.open(export_file_path);
        if (ofs.is_open()) {
            is_export = true;
        }
    }

    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "-------------------------------------------------------------------------------------");
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, std::string("Filename : " + file_path).c_str());
    std::string format_str{};
    if (format == Format::kONNX) {
        format_str = "onnx";
    }
    else if (format == Format::kPLAN) {
        format_str = "engine";
    }
    else {
        format_str = "unknown";
    }
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, std::string("Format   : " + format_str).c_str());
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "-------------------------------------------------------------------------------------");

    std::unique_ptr<IInspector> inspector{createInspector(file_path, format, gLogger)};

    if (!inspector->parsing()) {
        return EXIT_FAILURE;
    }
    inspector->summary(is_export ? ofs : std::cout);

    return EXIT_SUCCESS;
}