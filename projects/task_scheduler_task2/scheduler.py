from projects.task_scheduler_task2.graph import DependencyGraph

class Task:
    def __init__(self, task_id, priority=0):
        self.task_id = task_id
        self.priority = priority
        self.status = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED

class TaskScheduler:
    def __init__(self):
        self.tasks = {}
        self.graph = DependencyGraph()

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def add_dependency(self, task_id, dependency_id):
        self.graph.add_dependency(task_id, dependency_id)

    def get_runnable_tasks(self) -> list[Task]:
        """Returns tasks that are PENDING and all dependencies are COMPLETED."""
        runnable = []
        for task_id, task in self.tasks.items():
            if task.status != "PENDING":
                continue
            deps = self.graph.dependencies.get(task_id, [])
            all_completed = True
            for dep in deps:
                dep_task = self.tasks.get(dep)
                if not dep_task or dep_task.status != "COMPLETED":
                    all_completed = False
                    break
            if all_completed:
                runnable.append(task)

        runnable.sort(key=lambda t: t.priority, reverse=True)
        return runnable
