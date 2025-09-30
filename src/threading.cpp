#include "threading.hpp"

namespace threading {

ThreadPool::ThreadPool(size_t threads) : num_threads(threads) {
    workers.reserve(threads);
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this]() {
                        return stop.load() || !tasks.empty();
                    });

                    if (stop.load() && tasks.empty()) {
                        return;
                    }

                    if (!tasks.empty()) {
                        task = std::move(tasks.back());
                        tasks.pop_back();
                    }
                }

                if (task) {
                    active_tasks.fetch_add(1);
                    task();
                    active_tasks.fetch_sub(1);
                }
            }
        });
    }
}

void ThreadPool::wait() {
    while (active_tasks.load() > 0 || !tasks.empty()) {
        std::this_thread::yield();
    }
}

ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace threading
