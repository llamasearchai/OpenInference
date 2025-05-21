#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace openinference {
namespace runtime {

struct BatchConfig {
    int max_batch_size;
    int timeout_ms;
    bool enable_dynamic_size;
};

template<typename InputType, typename OutputType>
struct BatchRequest {
    InputType input;
    std::function<void(OutputType, bool)> callback;
    std::chrono::steady_clock::time_point arrival_time;
};

template<typename InputType, typename OutputType>
class BatchingManager {
public:
    explicit BatchingManager(const BatchConfig& config);
    ~BatchingManager();
    
    // Submit a request for batched processing
    void submit(const InputType& input, std::function<void(OutputType, bool)> callback);
    
    // Set the batch processing function
    void set_process_batch_fn(
        std::function<std::vector<OutputType>(const std::vector<InputType>&)> process_fn);
    
    // Start the batching thread
    void start();
    
    // Stop the batching thread
    void stop();
    
private:
    BatchConfig config_;
    std::function<std::vector<OutputType>(const std::vector<InputType>&)> process_batch_fn_;
    
    std::queue<BatchRequest<InputType, OutputType>> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::thread worker_thread_;
    std::atomic<bool> running_;
    
    void worker_loop();
    void process_batch();
};

} // namespace runtime
} // namespace openinference