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
    : ITaskSystem(num_threads)
    , mNumThreads(std::max(1, num_threads))
    , mThreads()
    , mCurrentRunnable(nullptr)
    , mCurrentNumTotalTasks(0)
    , mNextTaskId(0)
    , mCompletedTasks(0)
    , mIdleWorkers(0)
    , mHasActiveRun(false)
    , mShutdown(false) {
    mThreads.reserve(mNumThreads);
    for (int i = 0; i < mNumThreads; i++) {
        mThreads.emplace_back(&TaskSystemParallelThreadPoolSpinning::workerLoop, this);
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    mShutdown.store(true);
    mHasActiveRun.store(false);

    for (size_t i = 0; i < mThreads.size(); i++) {
        if (mThreads[i].joinable()) {
            mThreads[i].join();
        }
    }
}

void TaskSystemParallelThreadPoolSpinning::workerLoop() {
    bool is_idle = false;

    while (!mShutdown.load()) {
        if (!mHasActiveRun.load()) {
            if (!is_idle) {
                mIdleWorkers.fetch_add(1);
                is_idle = true;
            }
            std::this_thread::yield();
            continue;
        }

        if (is_idle) {
            mIdleWorkers.fetch_sub(1);
            is_idle = false;
        }

        IRunnable* runnable = mCurrentRunnable.load();
        int num_total_tasks = mCurrentNumTotalTasks.load();
        if (runnable == nullptr || num_total_tasks <= 0) {
            std::this_thread::yield();
            continue;
        }

        int task_id = mNextTaskId.fetch_add(1);
        if (task_id >= num_total_tasks) {
            std::this_thread::yield();
            continue;
        }

        runnable->runTask(task_id, num_total_tasks);
        int completed = mCompletedTasks.fetch_add(1) + 1;
        if (completed == num_total_tasks) {
            mHasActiveRun.store(false);
        }
    }

    if (is_idle) {
        mIdleWorkers.fetch_sub(1);
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    assert(runnable != nullptr);
    if (runnable == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> run_lock(mRunMutex);
    while (mHasActiveRun.load() || mIdleWorkers.load() != mNumThreads) {
        if (mShutdown.load()) {
            return;
        }
        std::this_thread::yield();
    }

    mNextTaskId.store(0);
    mCompletedTasks.store(0);
    mCurrentNumTotalTasks.store(num_total_tasks);
    mCurrentRunnable.store(runnable);
    mHasActiveRun.store(true);

    while (mHasActiveRun.load()) {
        if (mShutdown.load()) {
            break;
        }
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
    : ITaskSystem(num_threads)
    , mNumThreads(std::max(1, num_threads))
    , mThreads()
    , mCurrentRunnable(nullptr)
    , mCurrentNumTotalTasks(0)
    , mNextTaskId(0)
    , mCompletedTasks(0)
    , mHasActiveRun(false)
    , mShutdown(false) {
    mThreads.reserve(mNumThreads);
    for (int i = 0; i < mNumThreads; i++) {
        mThreads.emplace_back(&TaskSystemParallelThreadPoolSleeping::workerLoop, this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mShutdown = true;
        mHasActiveRun = false;
    }
    mWorkAvailableCV.notify_all();
    mLaunchDoneCV.notify_all();

    for (size_t i = 0; i < mThreads.size(); i++) {
        if (mThreads[i].joinable()) {
            mThreads[i].join();
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::workerLoop() {
    while (true) {
        std::unique_lock<std::mutex> lock(mMutex);
        mWorkAvailableCV.wait(lock, [this]() {
            return mShutdown || mHasActiveRun;
        });

        if (mShutdown) {
            return;
        }

        while (mHasActiveRun && !mShutdown) {
            if (mNextTaskId >= mCurrentNumTotalTasks) {
                mWorkAvailableCV.wait(lock, [this]() {
                    return mShutdown || !mHasActiveRun || (mNextTaskId < mCurrentNumTotalTasks);
                });
                if (mShutdown) {
                    return;
                }
                continue;
            }

            int task_id = mNextTaskId++;
            IRunnable* runnable = mCurrentRunnable;
            int num_total_tasks = mCurrentNumTotalTasks;

            lock.unlock();
            runnable->runTask(task_id, num_total_tasks);
            lock.lock();

            mCompletedTasks++;
            if (mCompletedTasks == mCurrentNumTotalTasks) {
                mHasActiveRun = false;
                mLaunchDoneCV.notify_one();
                mWorkAvailableCV.notify_all();
            }
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    assert(runnable != nullptr);
    if (runnable == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> run_lock(mRunMutex);
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mCurrentRunnable = runnable;
        mCurrentNumTotalTasks = num_total_tasks;
        mNextTaskId = 0;
        mCompletedTasks = 0;
        mHasActiveRun = true;
    }
    mWorkAvailableCV.notify_all();

    std::unique_lock<std::mutex> lock(mMutex);
    mLaunchDoneCV.wait(lock, [this]() {
        return !mHasActiveRun;
    });
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
