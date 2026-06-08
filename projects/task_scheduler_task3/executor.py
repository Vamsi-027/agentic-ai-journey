from projects.task_scheduler_task3.scheduler import TaskScheduler

class TaskExecutor:
    def __init__(self, scheduler: TaskScheduler):
        self.scheduler = scheduler

    def run_task(self, task_id, should_succeed=True):
        task = self.scheduler.tasks.get(task_id)
        if not task or task.status != "PENDING":
            return
        
        task.status = "RUNNING"
        if should_succeed:
            task.status = "COMPLETED"
        else:
            task.status = "FAILED"
            self._propagate_failure(task_id)

    def _propagate_failure(self, failed_task_id):
        """Finds downstream tasks that depend on the failed task and marks them as FAILED."""
        for task_id, task in self.scheduler.tasks.items():
            if task.status == "PENDING":
                deps = self.scheduler.graph.dependencies.get(task_id, [])
                if failed_task_id in deps:
                    task.status = "FAILED"
                    self._propagate_failure(task_id)
