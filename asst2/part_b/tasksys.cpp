#include "tasksys.h"
#include <algorithm>


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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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
    : ITaskSystem(num_threads),
      mNumThreads(num_threads),
      mWorkers(std::vector<std::thread>(num_threads)),
      mRunnableTasks(std::vector<TaskInfo>()),
      mWaitingTasks(std::vector<TaskInfo>()),
      mAllTasks(std::set<TaskID>()),
      mShutdown(false),
      mTaskIdgen(0),
      mRunnable(nullptr),
      mHasActiveTask(false),
      mNextTask(0),
      mNumTotalTasks(0),
      mFinishedTask(0) {
    for (int i = 0; i < num_threads; i++) {
        mWorkers[i] = std::thread(&TaskSystemParallelThreadPoolSleeping::workerLoop, this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    mShutdown.store(true);
    mCVWork.notify_all();
    for (int i = 0; i < mNumThreads; i++) {
        mWorkers[i].join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    runAsyncWithDeps(runnable, num_total_tasks, std::vector<TaskID>());
    sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable,
                                                              int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    TaskID taskId = mTaskIdgen.fetch_add(1);
    TaskInfo taskInfo{taskId, runnable, num_total_tasks, deps};
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (deps.empty()) {
            mRunnableTasks.push_back(taskInfo);
        } else {
            bool allDepsDone = true;
            for (TaskID dep : deps) {
                if (mAllTasks.find(dep) != mAllTasks.end()) {
                    allDepsDone = false;
                    break;
                }
            }
            if (allDepsDone) {
                mRunnableTasks.push_back(taskInfo);
            } else {
                mWaitingTasks.push_back(taskInfo);
            }
        }
        mAllTasks.insert(taskId);
        if (!mHasActiveTask.load()) {
            runNewTask();
        }
    }
    return taskId;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    {
        std::unique_lock<std::mutex> lock(mMutex);

        mCVDone.wait(lock, [&]() { return mAllTasks.empty() || mShutdown.load(); });

        if (mShutdown.load()) {
            return;
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::workerLoop() {
    TaskID lastSeenTaskId = -1;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mMutex);

            mCVWork.wait(lock, [&]() {
                return mShutdown.load() ||
                       (mHasActiveTask.load() && mCurrentTask.taskId != lastSeenTaskId);
            });

            if (mShutdown.load()) {
                return;
            }

            lastSeenTaskId = mCurrentTask.taskId;
        }

        while (true) {
            int taskId = mNextTask.fetch_add(1);

            if (taskId >= mNumTotalTasks.load()) {
                break;
            }

            mRunnable->runTask(taskId, mNumTotalTasks.load());

            int finished = mFinishedTask.fetch_add(1) + 1;
            if (finished == mNumTotalTasks.load()) {
                std::unique_lock<std::mutex> lock(mMutex);
                mAllTasks.erase(mCurrentTask.taskId);
                mHasActiveTask.store(false);
                activateTasks(mCurrentTask.taskId);
            }
        }
    }

    return;
}

// must be called with mMutex locked
void TaskSystemParallelThreadPoolSleeping::activateTasks(TaskID finishedTaskId) {
    for (auto task = mWaitingTasks.begin(); task != mWaitingTasks.end();) {
        auto index = std::find(task->deps.begin(), task->deps.end(), finishedTaskId);
        if (index != task->deps.end()) {
            task->deps.erase(index);
        }
        if (task->deps.empty()) {
            mRunnableTasks.push_back(*task);
            task = mWaitingTasks.erase(task);
        } else {
            task++;
        }
    }
    if (!mRunnableTasks.empty()) {
        runNewTask();
    } else {
        if (mAllTasks.empty()) {
            mCVDone.notify_all();
        }
    }
}

// must be called with mMutex locked
void TaskSystemParallelThreadPoolSleeping::runNewTask() {
    if (mRunnableTasks.empty()) {
        return;
    }

    TaskInfo task = mRunnableTasks.front();
    mRunnableTasks.erase(mRunnableTasks.begin());
    mCurrentTask = task;
    mRunnable = task.runnable;
    mNumTotalTasks = task.num_total_tasks;
    mNextTask.store(0);
    mFinishedTask.store(0);
    mHasActiveTask.store(true);
    mCVWork.notify_all();
}
