#pragma once
#include <string>
#include <memory>
#include <iomanip>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <NvInferVersion.h>

enum class Format
{
    kNONE,
    kONNX,
    kPLAN
};

class IInspector
{
public:
    IInspector(const std::string& file_path, nvinfer1::ILogger& logger)
    : file_path_(file_path), logger_(logger)
    {
    }
    virtual ~IInspector() = default;
    IInspector(const IInspector&) = delete;
    IInspector& operator=(const IInspector&) = delete;
    IInspector(IInspector&&) = delete;
    IInspector& operator=(IInspector&&) = delete;

    virtual bool parsing() = 0;
    virtual void summary(std::ostream& os) = 0;

protected:
    std::string file_path_;
    nvinfer1::ILogger& logger_;
};

class EngineInspector : public IInspector
{
public:
    EngineInspector(const std::string& file_path, nvinfer1::ILogger& logger)
    : IInspector(file_path, logger)
    {}
    virtual ~EngineInspector() {}

    bool parsing() override;
    void summary(std::ostream& os) override;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

class OnnxInspector : public IInspector
{
public:
    OnnxInspector(const std::string& file_path, nvinfer1::ILogger& logger)
    : IInspector(file_path, logger)
    {}
    virtual ~OnnxInspector() {}

    bool parsing() override;
    void summary(std::ostream& os) override;

private:
    std::unique_ptr<nvonnxparser::IParser> onnx_parser_;
    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::INetworkDefinition> network_;
};

struct BindingTensorInfo
{
    int idx{-1};
    bool is_input{false};
    std::string name;
    nvinfer1::Dims dim;
    nvinfer1::DataType type;
    nvinfer1::TensorFormat format;
    std::string format_desc;
};

struct NetworkLayerInfo
{
    std::string name;
    nvinfer1::LayerType type;
    std::vector<nvinfer1::Dims> dims;
    std::vector<int> io; // 0: none | 1: in | 2: out
};

struct EngineLayerInfo
{
    std::string name;
    std::string type;
    std::vector<std::string> dims;
};

template<typename T> inline
std::string getStr(const T& value)
{
    return std::to_string(value);
}

template<> inline
std::string getStr(const nvinfer1::Dims& value)
{
    std::string ret = "[";
    for (int i = 0; i < value.nbDims; i++) {
        ret += std::to_string(value.d[i]);
        if (i != value.nbDims - 1) {
            ret += ", ";
        }
    }
    ret += "]";
    return ret;
}

template<> inline
std::string getStr(const nvinfer1::DataType& value)
{
    using DataType = nvinfer1::DataType;
    if (value == DataType::kFLOAT) {
        return "FP32";
    }
    else if (value == DataType::kHALF) {
        return "FP16";
    }
    else if (value == DataType::kINT8) {
        return "INT8";
    }
    else if (value == DataType::kINT32) {
        return "INT32";
    }
    else if (value == DataType::kBOOL) {
        return "BOOL";
    }
#if (NV_TENSORRT_MAJOR >= 8) && (NV_TENSORRT_MINOR >= 5)
    else if (value == DataType::kUINT8) {
        return "UINT8";
    }
#endif
#if (NV_TENSORRT_MAJOR >= 8) && (NV_TENSORRT_MINOR >= 6)
    else if (value == DataType::kFP8) {
        return "FP8";
    }
#endif
    else {
        return "";
    }
}

template<> inline
std::string getStr(const nvinfer1::TensorFormat& value)
{
    using TensorFormat = nvinfer1::TensorFormat;
    if (value == TensorFormat::kLINEAR) {
        return "CHW";
    }
    else if (value == TensorFormat::kCHW2) {
        return "CHW2";
    }
    else if (value == TensorFormat::kHWC8) {
        return "HWC8";
    }
    else if (value == TensorFormat::kCHW4) {
        return "CHW4";
    }
    else if (value == TensorFormat::kCHW16) {
        return "CHW16";
    }
    else if (value == TensorFormat::kCHW32) {
        return "CHW32";
    }
    else if (value == TensorFormat::kDHWC8) {
        return "DHWC8";
    }
    else if (value == TensorFormat::kCDHW32) {
        return "CDHW32";
    }
    else if (value == TensorFormat::kHWC) {
        return "HWC";
    }
    else if (value == TensorFormat::kDLA_LINEAR) {
        return "CHW(DLA)";
    }
    else if (value == TensorFormat::kDLA_HWC4) {
        return "HWC4(DLA)";
    }
    else if (value == TensorFormat::kHWC16) {
        return "HWC16";
    }
#if (NV_TENSORRT_MAJOR >= 8) && (NV_TENSORRT_MINOR >= 6)
    else if (value == nvinfer1::TensorFormat::kDHWC) {
        return "DHWC";
    }
#endif
    else {
        return "";
    }
}

template<> inline
std::string getStr(const nvinfer1::LayerType& value)
{
    using LayerType = nvinfer1::LayerType;
    
    if (value == LayerType::kCONVOLUTION) {
        return "Convolution";
    }
    else if (value == LayerType::kFULLY_CONNECTED) {
        return "Fully Connected";
    }
    else if (value == LayerType::kACTIVATION) {
        return "Activation";
    }
    else if (value == LayerType::kPOOLING) {
        return "Pooling";
    }
    else if (value == LayerType::kLRN) {
        return "LRN";
    }
    else if (value == LayerType::kSCALE) {
        return "Scale";
    }
    else if (value == LayerType::kSOFTMAX) {
        return "Softmax";
    }
    else if (value == LayerType::kDECONVOLUTION) {
        return "Deconvolution";
    }
    else if (value == LayerType::kCONCATENATION) {
        return "Concatenation";
    }
    else if (value == LayerType::kELEMENTWISE) {
        return "Elementwise";
    }
    else if (value == LayerType::kPLUGIN) {
        return "Plugin";
    }
    else if (value == LayerType::kUNARY) {
        return "Unary";
    }
    else if (value == LayerType::kPADDING) {
        return "Padding";
    }
    else if (value == LayerType::kSHUFFLE) {
        return "Shuffle";
    }
    else if (value == LayerType::kREDUCE) {
        return "Reduce";
    }
    else if (value == LayerType::kTOPK) {
        return "TopK";
    }
    else if (value == LayerType::kGATHER) {
        return "Gather";
    }
    else if (value == LayerType::kMATRIX_MULTIPLY) {
        return "Matmul";
    }
    else if (value == LayerType::kRAGGED_SOFTMAX) {
        return "Ragged Softmax";
    }
    else if (value == LayerType::kCONSTANT) {
        return "Constant";
    }
    else if (value == LayerType::kRNN_V2) {
        return "RNN V2";
    }
    else if (value == LayerType::kIDENTITY) {
        return "Identity";
    }
    else if (value == LayerType::kPLUGIN_V2) {
        return "PluginV2";
    }
    else if (value == LayerType::kSLICE) {
        return "Slice";
    }
    else if (value == LayerType::kSHAPE) {
        return "Shape";
    }
    else if (value == LayerType::kPARAMETRIC_RELU) {
        return "Parametric ReLU";
    }
    else if (value == LayerType::kRESIZE) {
        return "Reisze";
    }
    else if (value == LayerType::kTRIP_LIMIT) {
        return "Loop Trip Limit";
    }
    else if (value == LayerType::kRECURRENCE) {
        return "Loop Recurrence";
    }
    else if (value == LayerType::kITERATOR) {
        return "Loop Iterator";
    }
    else if (value == LayerType::kLOOP_OUTPUT) {
        return "Loop Output";
    }
    else if (value == LayerType::kSELECT) {
        return "Select";
    }
    else if (value == LayerType::kFILL) {
        return "Fill";
    }
    else if (value == LayerType::kQUANTIZE) {
        return "Quantize";
    }
    else if (value == LayerType::kDECONVOLUTION) {
        return "Dequantize";
    }
    else if (value == LayerType::kCONDITION) {
        return "Condition";
    }
    else if (value == LayerType::kCONDITIONAL_INPUT) {
        return "Conditional Input";
    }
    else if (value == LayerType::kCONDITIONAL_OUTPUT) {
        return "Conditional Output";
    }
    else if (value == LayerType::kSCATTER) {
        return "Scatter";
    }
    else if (value == LayerType::kEINSUM) {
        return "Einsum";
    }
    else if (value == LayerType::kASSERTION) {
        return "Assertion";
    }
#if (NV_TENSORRT_MAJOR >= 8) && (NV_TENSORRT_MINOR >= 5)
    else if (value == LayerType::kONE_HOT) {
        return "OneHot";
    }
    else if (value == LayerType::kNON_ZERO) {
        return "NonZero";
    }
    else if (value == LayerType::kGRID_SAMPLE) {
        return "Grid Sample";
    }
    else if (value == LayerType::kNMS) {
        return "NMS";
    }
#endif
#if (NV_TENSORRT_MAJOR >= 8) && (NV_TENSORRT_MINOR >= 5)
    else if (value == LayerType::kREVERSE_SEQUENCE) {
        return "Reverse Sequence";
    }
    else if (value == LayerType::kNORMALIZATION) {
        return "Normalization";
    }
    else if (value == LayerType::kCAST) {
        return "Cast";
    }
#endif
    else {
        return "";
    }
}

template<> inline
std::string getStr(const nvinfer1::ProfilingVerbosity& value)
{
    using PV = nvinfer1::ProfilingVerbosity;
    if (value == PV::kLAYER_NAMES_ONLY) {
        return "layer_names_only";
    }
    else if (value == PV::kDETAILED) {
        return "detailed";
    }
    else {
        return "none";
    }
}

inline
std::ostream& operator<<(std::ostream& os, const std::vector<BindingTensorInfo>& tensor_info)
{
    const int idx_width = 5;
    const int name_width = 30;
    const int io_width = 6;
    const int dims_width = 20;
    const int type_width = 10;
    const int format_width = 15;
    const int total_width = idx_width + name_width + io_width + dims_width + type_width + format_width;

    int num_binding = tensor_info.size();
    auto get_str = [](bool is_input) -> std::string {
        if (is_input) {
            return "(in)";
        }
        else {
            return "(out)";
        }
    };

    os << std::setw(total_width) << std::setfill('-') << "-" << "\n";
    os << std::setw(idx_width) << std::setfill(' ') << std::left << "Idx"
        << std::setw(name_width) << "Tensor Name"
        << std::setw(io_width) << " "
        << std::setw(dims_width) << "Shape"
        << std::setw(type_width) << "Type"
        << std::setw(format_width) << "Format" << "\n";
    os << std::setw(total_width) << std::setfill('=') << "=" << std::setfill(' ') << "\n";
    for (int i = 0; i < num_binding; i++) {
        auto& t = tensor_info[i];
        std::string name{t.name};
        if (name.length() > name_width- 1) {
            name.resize(name_width - 1);
            name.replace(name.length() - 3, 3, "...");
        }
        os << std::left << std::setw(idx_width) << t.idx
            << std::setw(name_width) << name << std::setw(io_width) << get_str(t.is_input)
            << std::setw(dims_width) << getStr(t.dim)
            << std::setw(type_width) << getStr(t.type)
            << std::setw(format_width) << getStr(t.format) << "\n";
    }
    os << std::setw(total_width) << std::setfill('-') << "-" << std::setfill(' ') << "\n";
    return os;
}

inline
std::ostream& operator<<(std::ostream& os, const std::vector<NetworkLayerInfo>& layer_info)
{
    const int no_width = 5;
    const int name_width = 50;
    const int type_width = 25;
    const int dims_width = 20;
    const int io_width = 6;
    const int total_width = no_width + name_width + type_width + dims_width + io_width;

    int num_layer = layer_info.size();
    auto get_str = [](int io) -> std::string {
        if (io == 1) {
            return "(in)";
        }
        else if (io == 2) {
            return "(out)";
        }
        else {
            return "";
        }
    };

    os << std::setw(total_width) << std::setfill('-') << "-" << "\n";
    os << std::setw(no_width) << std::setfill(' ') << std::left << "No"
        << std::setw(name_width) << "Layer Name"
        << std::setw(type_width) << "Type"
        << std::setw(dims_width) << "Output Shape" << "\n";
    os << std::setw(total_width) << std::setfill('=') << "=" << std::setfill(' ') << "\n";
    for (int i = 0; i < num_layer; i++) {
        auto& l = layer_info[i];
        std::string name(l.name);
        if (name.length() > name_width - 1) {
            name.resize(name_width - 1);
            name.replace(name.length() - 3, 3, "...");
        }
        os << std::left << std::setw(no_width) << (i+1)
            << std::setw(name_width) << name
            << std::setw(type_width) << getStr(l.type)
            << std::setw(dims_width) << getStr(l.dims[0])
            << std::setw(io_width) << std::right << get_str(l.io[0]) << std::left << "\n";
        const int num_output = l.dims.size();
        if (num_output > 1) {
            for (int k = 1; k < num_output; k++) {
                os << std::setw(no_width + name_width + type_width) << " "
                    << std::setw(dims_width) << getStr(l.dims[k])
                    << std::setw(io_width) << std::right << get_str(l.io[k]) << std::left << "\n";
            }
        }
    }
    os << std::setw(total_width) << std::setfill('-') << "-" << std::setfill(' ') << "\n";
    return os;
}

inline
std::ostream& operator<<(std::ostream& os, const std::vector<EngineLayerInfo>& layer_info)
{
    const int no_width = 5;
    const int name_width = 120;
    const int type_width = 30;
    const int dims_width = 20;
    const int total_width = no_width + name_width + type_width + dims_width;

    int num_layer = layer_info.size();


    os << std::setw(total_width) << std::setfill('-') << "-" << "\n";
    os << std::setw(no_width) << std::setfill(' ') << std::left << "No"
        << std::setw(name_width) << "Layer Name"
        << std::setw(type_width) << "Type"
        << std::setw(dims_width) << "Output Shape" << "\n";
    os << std::setw(total_width) << std::setfill('=') << "=" << std::setfill(' ') << "\n";
    for (int i = 0; i < num_layer; i++) {
        auto& l = layer_info[i];
        std::string name(l.name);
        if (name.length() > name_width - 1) {
            name.resize(name_width - 1);
            name.replace(name.length() - 3, 3, "...");
        }
        os << std::left << std::setw(no_width) << (i+1)
            << std::setw(name_width) << name;
        if (l.type.length() > 0) {
            os << std::setw(type_width) << l.type;
        }
        const int num_output = l.dims.size();
        if (num_output) {
            os << std::setw(dims_width) << l.dims[0];
        }
        os  << "\n";
        if (num_output > 1) {
            for (int k = 1; k < num_output; k++) {
                os << std::setw(no_width + name_width + type_width) << " "
                    << std::setw(dims_width) << l.dims[k] << "\n";
            }
        }
    }
    os << std::setw(total_width) << std::setfill('-') << "-" << std::setfill(' ') << "\n";
    return os;
}

inline
IInspector* createInspector(const std::string& file_path, Format file_format, nvinfer1::ILogger& logger)
{
    IInspector* inspector{nullptr};
    if (file_format == Format::kONNX) {
        inspector = new OnnxInspector(file_path, logger);
    }
    else if (file_format == Format::kPLAN) {
        inspector = new EngineInspector(file_path, logger);
    }
    else {
        inspector = nullptr;
    }

    return inspector;
}