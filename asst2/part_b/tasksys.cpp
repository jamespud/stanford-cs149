#include "tasksys.h"
#include <algorithm>
#include <cstdio>


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
    printf("[LAUNCH] runAsyncWithDeps called, launchId=%d, num_tasks=%d, deps.size()=%zu\n",
           taskId, num_total_tasks, deps.size());
    TaskInfo taskInfo{taskId, runnable, num_total_tasks, deps};
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (deps.empty()) {
            printf("[LAUNCH] No deps, adding to runnable tasks\n");
            mRunnableTasks.push_back(taskInfo);
        } else {
            // check deps is done or not
            bool allDepsDone = true;
            for (TaskID dep : deps) {
                if (mAllTasks.find(dep) != mAllTasks.end()) {
                    allDepsDone = false;
                    break;
                }
            }
            if (allDepsDone) {
                printf("[LAUNCH] All deps done, adding to runnable tasks\n");
                mRunnableTasks.push_back(taskInfo);
            } else {
                printf("[LAUNCH] Adding to waiting tasks\n");
                mWaitingTasks.push_back(taskInfo);
            }
        }
        mAllTasks.insert(taskId);
        printf("[LAUNCH] After insert, mAllTasks.size()=%zu, mRunnable=%p\n",
               mAllTasks.size(), mRunnable);
        if (!mHasActiveTask.load()) {
            printf("[LAUNCH] No current task, calling runNewTask()\n");
            runNewTask();
        }
    }
    mCVWork.notify_all();
    return taskId;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    printf("[SYNC] Entering sync(), mAllTasks.size()=%zu\n", mAllTasks.size());
        {
            std::unique_lock<std::mutex> lock(mMutex);

            printf("[SYNC] Waiting for mAllTasks to be empty, current size=%zu\n", mAllTasks.size());
            mCVWork.wait(lock, [&]() { return mAllTasks.empty() || mShutdown.load(); });

            printf("[SYNC] Woke up, mAllTasks.size()=%zu, shutdown=%d\n", mAllTasks.size(), mShutdown.load());
            if (mShutdown.load()) {
                return;
            }
           
        }
    
    printf("[SYNC] Exiting sync()\n");
}

void TaskSystemParallelThreadPoolSleeping::workerLoop() {
    printf("[WORKER] Worker thread started\n");
    int taskId = -1;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mMutex);

            printf("[WORKER] Waiting for work, =%zu\n", mRunnableTasks.size());
            mCVWork.wait(lock, [&]() { return mHasActiveTask.load() || mShutdown.load(); });

            if (mShutdown.load()) {
                printf("[WORKER] Worker shutting down\n");
                return;
            }
        }

        printf("[WORKER] Starting task execution loop\n");
        while (true) {
            taskId = mNextTask.fetch_add(1);

            if (taskId >= mNumTotalTasks.load()) {
                printf("[WORKER] Task %d >= total %d, breaking\n", taskId, mNumTotalTasks.load());
                break;
            }

            mRunnable->runTask(taskId, mNumTotalTasks.load());

            int finished = mFinishedTask.fetch_add(1) + 1;
            printf("[WORKER] Task %d finished, total finished=%d/%d\n", taskId, finished, mNumTotalTasks.load());
            if (finished == mNumTotalTasks.load()) {
                printf("[WORKER] All tasks done, calling activateTasks for launch %d\n", mCurrentTask.taskId);
                std::unique_lock<std::mutex> lock(mMutex);
                mAllTasks.erase(mCurrentTask.taskId);
                mHasActiveTask.store(false);
                printf("[WORKER] After erase, mAllTasks.size()=%zu\n", mAllTasks.size());
                activateTasks(mCurrentTask.taskId);
            }
        }
    }

    return;
}

// must be called with mMutex locked
void TaskSystemParallelThreadPoolSleeping::activateTasks(TaskID finishedTaskId) {
    printf("[ACTIVATE] activateTasks called for finished launch %d, mWaitingTasks.size()=%zu, mRunnableTasks.size()=%zu\n",
           finishedTaskId, mWaitingTasks.size(), mRunnableTasks.size());
    for (auto task = mWaitingTasks.begin(); task != mWaitingTasks.end();) {
        auto index = std::find(task->deps.begin(), task->deps.end(), finishedTaskId);
        if (index != task->deps.end()) {
            task->deps.erase(index);
        }
        if (task->deps.empty()) {
            printf("[ACTIVATE] Moving launch %d from waiting to runnable\n", task->taskId);
            mRunnableTasks.push_back(*task);
            task = mWaitingTasks.erase(task);
        } else {
            task++;
        }
    }
    printf("[ACTIVATE] After processing, mAllTasks.size()=%zu, mRunnableTasks.size()=%zu\n",
           mAllTasks.size(), mRunnableTasks.size());
    if (!mRunnableTasks.empty()) {
        printf("[ACTIVATE] Calling runNewTask()\n");
        runNewTask();
    } else {
        printf("[ACTIVATE] mAllTasks is empty, notifying all\n");
        mCVWork.notify_all();
    }
}

// must be called with mMutex locked
void TaskSystemParallelThreadPoolSleeping::runNewTask() {
    printf("[RUNNEW] runNewTask called, mRunnableTasks.size()=%zu\n", mRunnableTasks.size());
    if (mRunnableTasks.empty()) {
        printf("[RUNNEW] No runnable tasks, returning\n");
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
    printf("[RUNNEW] Loaded launch %d with %d tasks, notifying workers\n", task.taskId, task.num_total_tasks);
    mCVWork.notify_all();
}
