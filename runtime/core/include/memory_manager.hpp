#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace openinference {
namespace runtime {

class MemoryBlock {
public:
    MemoryBlock(void* ptr, size_t size);
    ~MemoryBlock();
    
    void* data();
    size_t size() const;
    
private:
    void* ptr_;
    size_t size_;
};

class MemoryManager {
public:
    enum class AllocationType {
        HOST,
        DEVICE,
        PINNED,
        UNIFIED
    };
    
    MemoryManager(int device_id = 0);
    ~MemoryManager();
    
    // Allocate memory block
    std::shared_ptr<MemoryBlock> allocate(size_t size, AllocationType type);
    
    // Free a specific allocation
    void free(void* ptr);
    
    // Free all allocations
    void clear();
    
    // Get current memory usage stats
    std::unordered_map<AllocationType, size_t> get_memory_stats() const;
    
private:
    int device_id_;
    mutable std::mutex mutex_;
    std::unordered_map<void*, std::shared_ptr<MemoryBlock>> allocations_;
    std::unordered_map<AllocationType, size_t> allocated_bytes_;
    
    void* allocate_memory(size_t size, AllocationType type);
    void free_memory(void* ptr, AllocationType type);
};

} // namespace runtime
} // namespace openinference