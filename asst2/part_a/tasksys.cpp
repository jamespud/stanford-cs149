#include "tasksys.h"

#include <algorithm>
#include <cassert>


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    this->mNumThreads = std::max(1, num_threads);
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}


void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    assert(runnable != nullptr);
    if (runnable == nullptr) {
        return;
    }

    const int worker_count = std::min(mNumThreads, num_total_tasks);
    std::atomic<int> next_task_id(0);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    for (int i = 0; i < worker_count; i++) {
        threads.emplace_back([runnable, num_total_tasks, &next_task_id]() {
            while (true) {
                int task_id = next_task_id.fetch_add(1);
                if (task_id >= num_total_tasks) {
                    break;
                }
                runnable->runTask(task_id, num_total_tasks);
            }
        });
    }

    for (size_t i = 0; i < threads.size(); i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }

}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads)
    : ITaskSystem(num_threads) {
    mNumThreads = std::max(1, num_threads);

    mShutdown.store(false);
    mHasWork.store(false);

    mRunnable = nullptr;
    mNumTotalTasks = 0;

    mNextTask.store(0);
    mFinishedTask.store(0);

    for (int i = 0; i < mNumThreads; i++) {
        mWorkers.emplace_back(&TaskSystemParallelThreadPoolSpinning::workerLoop, this);
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    mShutdown.store(true, std::memory_order_seq_cst);

    for (auto& t : mWorkers) {
        t.join();
    }
}

void TaskSystemParallelThreadPoolSpinning::workerLoop() {
    while (!mShutdown.load(std::memory_order_seq_cst)) {
        if (!mHasWork.load(std::memory_order_acquire)) {
            std::this_thread::yield();
            continue;
        }

        int taskId = mNextTask.fetch_add(1);

        if (taskId >= mNumTotalTasks) {
            std::this_thread::yield();
            continue;
        }

        mRunnable->runTask(taskId, mNumTotalTasks);

        int finished = mFinishedTask.fetch_add(1) + 1;

        if (finished == mNumTotalTasks) {
            mHasWork.store(false, std::memory_order_release);
        }
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    mRunnable = runnable;
    mNumTotalTasks = num_total_tasks;

    mNextTask.store(0);
    mFinishedTask.store(0);

    /*
     * release:
     * 保证 runnable / num_total_tasks
     * 在 worker 看到 hasWork=true 前可见
     */
    mHasWork.store(true, std::memory_order_release);

    while (mHasWork.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads)
    : ITaskSystem(num_threads) {
    mNumThreads = std::max(1, num_threads);

    mShutdown.store(false);

    mRunnable = nullptr;
    mNumTotalTasks = 0;

    mNextTask.store(0);
    mFinishedTask.store(0);

    mHasWork = false;

    for (int i = 0; i < mNumThreads; i++) {
        mWorkers.emplace_back(&TaskSystemParallelThreadPoolSleeping::workerLoop, this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mShutdown.store(true);
        mHasWork = true;
    }

    mCVWork.notify_all();

    for (auto& t : mWorkers) {
        t.join();
    }
}

void TaskSystemParallelThreadPoolSleeping::workerLoop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mMutex);

            mCVWork.wait(lock, [&]() { return mHasWork || mShutdown.load(); });

            if (mShutdown.load()) {
                return;
            }
        }

        while (true) {
            int taskId = mNextTask.fetch_add(1);

            if (taskId >= mNumTotalTasks) {
                break;
            }

            mRunnable->runTask(taskId, mNumTotalTasks);

            int finished = mFinishedTask.fetch_add(1) + 1;

            if (finished == mNumTotalTasks) {
                std::unique_lock<std::mutex> lock(mMutex);
                mHasWork = false;
                mCVDone.notify_one();
            }
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(mMutex);

        mRunnable = runnable;
        mNumTotalTasks = num_total_tasks;

        mNextTask.store(0);
        mFinishedTask.store(0);

        mHasWork = true;
    }

    mCVWork.notify_all();

    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCVDone.wait(lock, [&]() { return !mHasWork; });
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
