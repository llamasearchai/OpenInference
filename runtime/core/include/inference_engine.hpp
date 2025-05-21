#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <cstdint>

#include "memory_manager.hpp"
#include "batching.hpp"

namespace openinference {
namespace runtime {

enum class DeviceType {
    CPU,
    CUDA,
    METAL,
    VULKAN
};

enum class ModelType {
    PYTORCH,
    ONNX,
    TENSORRT,
    CUSTOM
};

struct InferenceOptions {
    DeviceType device_type = DeviceType::CPU;
    int device_id = 0;
    bool use_fp16 = false;
    bool use_bf16 = false;
    bool use_int8 = false;
    bool enable_graph_optimization = true;
    bool enable_dynamic_batching = true;
    int max_batch_size = 64;
    int batch_timeout_ms = 10;
    std::string cache_dir = "";
};

class InferenceEngine {
public:
    InferenceEngine(const InferenceOptions& options);
    ~InferenceEngine();

    // Disable copy and move
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = delete;
    InferenceEngine& operator=(InferenceEngine&&) = delete;

    // Load a model from disk
    bool load_model(const std::string& model_path, ModelType model_type);
    
    // Run inference on input data
    std::vector<float> infer(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape);
    
    // Async inference with callback
    void infer_async(const std::vector<float>& input_data, 
                    const std::vector<int64_t>& input_shape,
                    std::function<void(std::vector<float>, bool)> callback);
    
    // Get model metadata
    std::unordered_map<std::string, std::string> get_model_metadata() const;
    
    // Get inference statistics 
    std::unordered_map<std::string, double> get_statistics() const;
    
    // Warm up the model with random inputs
    void warmup(int num_runs = 5);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Memory manager for efficient tensor allocation
    std::unique_ptr<MemoryManager> memory_manager_;
    
    // Batching manager for efficient request batching
    std::unique_ptr<BatchingManager> batching_manager_;
    
    // Internal methods
    void initialize_runtime();
    void optimize_model();
    void setup_profiling();
};

} // namespace runtime
} // namespace openinference