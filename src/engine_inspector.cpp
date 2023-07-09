#include <inspector.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
class EngineInfoJsonParser
{
    using PV = nvinfer1::ProfilingVerbosity;
public:
    explicit EngineInfoJsonParser(const char* raw_json, const PV pv) {
        std::string json{raw_json};
        if (pv == PV::kDETAILED) { // detailed
            int idx = 0;
            auto get_str = [&json](int& idx) {
                while (json[idx] != '"') idx++;
                idx++;
                int end_idx = idx;
                while (json[end_idx] != '"') end_idx++;
                std::string key = json.substr(idx, end_idx - idx);
                idx = end_idx + 1;
                return key;
            };
            while (idx < json.length()) {
                if (json[idx] == '{') {
                    idx++;
                    if (json.substr(idx + 1, 6).compare("Layers") == 0) {
                        // get layer info
                        while (idx < json.length()) {
                            if (json[idx] == '{') {
                                EngineLayerInfo layer_info;
                                while (get_str(idx).compare("Name") != 0);
                                layer_info.name = get_str(idx);

                                while (get_str(idx).compare("LayerType") != 0);
                                layer_info.type = get_str(idx);

                                while (get_str(idx).compare("Outputs") != 0);
                                while (json[idx] != ']') {
                                    while (get_str(idx).compare("Dimensions") != 0);
                                    idx += 2;
                                    int end_idx = idx;
                                    while (json[end_idx] != ']') end_idx++;
                                    layer_info.dims.push_back(json.substr(idx, end_idx - idx + 1));
                                    idx = end_idx + 1;
                                    while (json[idx] != '}') idx++;
                                    idx++;
                                }
                                while (json[idx] != '}' || json[idx + 1] != ',' || json[idx + 2] != '{') {
                                    idx++;
                                    if (idx + 2 >= json.length()) break;
                                }
                                layer_info_.push_back(layer_info);
                                idx++;
                            }
                            idx++;
                        }
                    }
                }
                idx++;
            }
        }
        else { // layer-name-only
            int idx = 0;
            std::string layer_name;
            while (idx < json.length()) {
                if (json[idx] == '[') {
                    idx++;
                    while (json[idx] != ']') {
                        if (json[idx] == '"') {
                            int find_idx = idx + 1;
                            while (json[find_idx] != '"') find_idx++;
                            layer_name = json.substr(idx + 1, find_idx - idx - 1);
                            layer_info_.push_back({layer_name});
                            idx = find_idx;
                        }
                        idx++;
                    }
                }
                idx++;
            }
        }
    }

    const std::vector<EngineLayerInfo>& getLayerInfo() const {
        return layer_info_;
    }

private:
    std::vector<EngineLayerInfo> layer_info_;
};

bool EngineInspector::parsing()
{
    using Severity = nvinfer1::ILogger::Severity;
    std::ifstream engine_file(file_path_, std::ios::binary);
    if (!engine_file.good()) {
        std::string err_msg = "Error: cannot parse engine file: " + file_path_;
        logger_.log(Severity::kERROR, err_msg.c_str());
        return false;   
    }
    engine_file.seekg(0, std::ifstream::end);
    int64_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engine_blobs(file_size);
    engine_file.read(reinterpret_cast<char*>(engine_blobs.data()), file_size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_blobs.data(), file_size));
    if (!engine_) {
        logger_.log(Severity::kERROR, "Error: cannot create an engine");
        return false;
    }
    logger_.log(Severity::kINFO, "Parsing is done");

    return true;
}

void EngineInspector::summary(std::ostream& os)
{
    using Severity = nvinfer1::ILogger::Severity;
    std::stringstream ss;
    // query layer infomation (WIP)
    auto profiling_verbosity = engine_->getProfilingVerbosity();
    if (profiling_verbosity != nvinfer1::ProfilingVerbosity::kNONE) {
        auto inspector = engine_->createEngineInspector();
        const int num_layer = engine_->getNbLayers();
        const char* engine_info_json = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
        auto json_parser = EngineInfoJsonParser(engine_info_json, profiling_verbosity);
        ss.str("");
        ss << json_parser.getLayerInfo();
        os << "> Engine Layer Summary:\n";
        os << ss.str();
    }
    else {
        logger_.log(Severity::kWARNING, "Skip layer summary (Profiling Verbosity of the Engine: NONE)");
    }

    // query binding tensor infomation
    int num_binding = engine_->getNbBindings();
    std::vector<BindingTensorInfo> tensor_info(num_binding);
    for (int i = 0; i < num_binding; i++) {
        tensor_info[i].idx = i;
        tensor_info[i].is_input = engine_->bindingIsInput(i);
        tensor_info[i].name = std::string(engine_->getBindingName(i));
        tensor_info[i].dim = engine_->getBindingDimensions(i);
        tensor_info[i].type = engine_->getBindingDataType(i);
        tensor_info[i].format = engine_->getBindingFormat(i);
        tensor_info[i].format_desc = std::string(engine_->getBindingFormatDesc(i));
    }
    ss.str("");
    ss << tensor_info;
    
    os << "> Engine Binding Tensor Summary:\n";
    os << ss.str();
}