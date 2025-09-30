#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace threading {

// Thread pool for reusing worker threads
class ThreadPool {
  private:
    std::vector<std::thread> workers;
    std::vector<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<size_t> active_tasks{0};
    std::atomic<bool> stop{false};
    size_t num_threads;

  public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    // Enqueue a task to be executed by the thread pool
    template <typename F> void enqueue(F &&f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace_back(std::forward<F>(f));
        }
        condition.notify_one();
    }

    // Wait for all tasks to complete
    void wait();

    // Get number of threads in pool
    size_t thread_count() const { return num_threads; }

    // Disable copying
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
};

} // namespace threading
