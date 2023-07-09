#include <inspector.h>
#include <iostream>

bool OnnxInspector::parsing()
{
    builder_.reset(nvinfer1::createInferBuilder(logger_));
    network_.reset(builder_->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

    onnx_parser_.reset(nvonnxparser::createParser(*network_, logger_));
    bool parsed = onnx_parser_->parseFromFile(file_path_.c_str(), 3);
    if (!parsed) {
        std::string err_msg = "Error: cannot parse onnx file: " + file_path_;
        logger_.log(nvinfer1::ILogger::Severity::kERROR, err_msg.c_str());
        return false;
    }
    logger_.log(nvinfer1::ILogger::Severity::kINFO, "Parsing is done");

    return true;
}

void OnnxInspector::summary(std::ostream& os)
{
    using Severity = nvinfer1::ILogger::Severity;
    std::stringstream ss;
    // query layer infomation
    // layer name | type | output shape | in/out
    const int num_layer = network_->getNbLayers();
    std::vector<NetworkLayerInfo> layer_info(num_layer);
    for (int i = 0; i < num_layer; i++) {
        auto& layer = *network_->getLayer(i);
        layer_info[i].name = layer.getName();
        layer_info[i].type = layer.getType();

        const int num_output = layer.getNbOutputs();
        layer_info[i].dims.resize(num_output);
        layer_info[i].io.resize(num_output);
        for (int j = 0; j < num_output; j++) {
            auto& tensor = *layer.getOutput(j);
            layer_info[i].dims[j] = tensor.getDimensions();
            if (tensor.isNetworkInput()) {
                layer_info[i].io[j] = 1;
            }
            else if (tensor.isNetworkOutput()) {
                layer_info[i].io[j] = 2;
            }
            else {
                layer_info[i].io[j] = 0;
            }
        }
    }
    ss.str("");
    ss << layer_info;
    os << "> NetworkDefinition Layer Summary:\n";
    os << ss.str();

    // query binding tensor infomation
    const int num_input = network_->getNbInputs();
    const int num_output = network_->getNbOutputs();
    std::vector<BindingTensorInfo> tensor_info(num_input + num_output);
    for (int i = 0; i < num_input; i++) {
        auto& tensor = *network_->getInput(i);
        tensor_info[i].idx = i;
        tensor_info[i].is_input = true;
        tensor_info[i].name = std::string(tensor.getName());
        tensor_info[i].dim = tensor.getDimensions();
        tensor_info[i].type = tensor.getType();
        uint32_t allowed_formats = tensor.getAllowedFormats();
        int allowed_format_cnt = 0;
        for (int k = 0; k < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); k++) {
            if (allowed_formats & (1U << k)) {
                allowed_format_cnt++;
            }
            if (allowed_format_cnt == 1) {
                tensor_info[i].format = static_cast<nvinfer1::TensorFormat>(k);
            }
        }
        if (allowed_format_cnt > 1) {
            ss.str("");
            ss << "Input " << i << " has multiple allowed formats: ";
            for (int k = 0, cnt = 0; k < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); k++) {
                if (allowed_formats & (1U << k)) {
                    ss << getStr(static_cast<nvinfer1::TensorFormat>(k));
                    cnt++;
                }
                if (cnt < allowed_format_cnt) {
                    ss << " | ";
                }
            }
            logger_.log(Severity::kWARNING, ss.str().c_str());
        }
        tensor_info[i].format_desc = "";
    }
    for (int i = num_input; i < num_input + num_output; i++) {
        auto& tensor = *network_->getOutput(i - num_input);
        tensor_info[i].idx = i - num_input;
        tensor_info[i].is_input = false;
        tensor_info[i].name = std::string(tensor.getName());
        tensor_info[i].dim = tensor.getDimensions();
        tensor_info[i].type = tensor.getType();
        uint32_t allowed_formats = tensor.getAllowedFormats();
        int allowed_format_cnt = 0;
        for (int k = 0; k < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); k++) {
            if (allowed_formats & (1U << k)) {
                allowed_format_cnt++;
            }
            if (allowed_format_cnt == 1) {
                tensor_info[i].format = static_cast<nvinfer1::TensorFormat>(k);
            }
        }
        if (allowed_format_cnt > 1) {
            ss.str("");
            ss << "Output " << tensor_info[i].idx << " has multiple allowed formats: ";
            for (int k = 0, cnt = 0; k < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); k++) {
                if (allowed_formats & (1U << k)) {
                    ss << getStr(static_cast<nvinfer1::TensorFormat>(k));
                    cnt++;
                }
                if (cnt < allowed_format_cnt) {
                    ss << " | ";
                }
            }
            logger_.log(Severity::kWARNING, ss.str().c_str());
        }
        tensor_info[i].format_desc = "";
    }
    ss.str("");
    ss << tensor_info;
    
    os << "> Expected Binding Tensor Summary:\n";
    os << ss.str();
}