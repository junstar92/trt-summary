#include <inspector.h>
#include <json.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

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
    // query layer infomation
    auto profiling_verbosity = engine_->getProfilingVerbosity();
    if (profiling_verbosity != nvinfer1::ProfilingVerbosity::kNONE) {
        auto inspector = engine_->createEngineInspector();
        auto json_contents = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
        Json::Reader reader;
        Json::Value root;

        bool success = reader.parse(json_contents, root, false);
        if (success) {
            Json::Value layers = root["Layers"];
            auto num_of_layer = layers.size();
            std::vector<EngineLayerInfo> engine_layer_info(num_of_layer);
            auto dims_to_str = [](auto& dims_object) {
                std::string ret = "[";
                auto size = dims_object.size();
                for (int i = 0; i < size; i++) {
                    ret += std::to_string(dims_object[i].asInt()) + ",";
                }
                ret[ret.length() - 1] = ']';
                return ret;
            };
            for (unsigned int i = 0; i < num_of_layer; i++) {
                auto& layer = layers[i];
                if (layer.isObject()) {
                    engine_layer_info[i].name = layer["Name"].asString();
                    engine_layer_info[i].type = layer["LayerType"].asString();
                    auto num_of_outputs = layer["Outputs"].size();
                    for (unsigned int j = 0; j < num_of_outputs; j++) {
                        engine_layer_info[i].dims.push_back(dims_to_str(layer["Outputs"][j]["Dimensions"]));
                    }
                }
                else {
                    engine_layer_info[i].name = layer.asString();
                }
            }
            ss.str("");
            ss << engine_layer_info;
            os << "> Engine Layer Summary:\n";
            os << ss.str();
        }
        else {
            logger_.log(Severity::kERROR, "Failed to parse json contents.");
        }
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