import pytest
from projects.task_scheduler_task1.graph import DependencyGraph
from projects.task_scheduler_task1.scheduler import Task, TaskScheduler
from projects.task_scheduler_task1.executor import TaskExecutor

def test_dependency_ordering():
    graph = DependencyGraph()
    graph.add_dependency("A", "B")
    order = graph.find_order(["A", "B"])
    assert order == ["B", "A"]

def test_priority_execution():
    scheduler = TaskScheduler()
    task_low = Task("low_priority", priority=1)
    task_high = Task("high_priority", priority=10)
    
    scheduler.add_task(task_low)
    scheduler.add_task(task_high)
    
    runnable = scheduler.get_runnable_tasks()
    assert runnable[0].task_id == "high_priority"

def test_dependency_failure_propagation():
    scheduler = TaskScheduler()
    task_a = Task("A")
    task_b = Task("B")
    
    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_dependency("B", "A")
    
    executor = TaskExecutor(scheduler)
    executor.run_task("A", should_succeed=False)
    
    assert scheduler.tasks["B"].status == "FAILED"
