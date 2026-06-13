import os
import sys
import time
import sqlite3
import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.core.llm import get_llm_client, LLMProvider, ClaudeModel, OpenAIClient
from src.core.retrieval.vector_store import VectorStore
from src.core.retrieval.rag_pipeline import RAGPipeline
from src.core.agent.react import ReActAgent
from src.core.database import DEFAULT_DB_PATH
from src.core.llm.tools import (
    WRITE_FILE_TOOL, write_file,
    READ_FILE_TOOL, read_file,
    RUN_PYTHON_TOOL, run_python,
    RUN_TESTS_TOOL, run_tests,
    EDIT_FILE_TOOL, edit_file
)

# ==============================================================================
# Buggy Code Contents (Fixtures)
# ==============================================================================

BUGGY_CODES = {
    "task1": """class DependencyGraph:
    def __init__(self):
        self.dependencies = {}  # task_id -> list of dependencies

    def add_dependency(self, task_id, dependency_id):
        if task_id not in self.dependencies:
            self.dependencies[task_id] = []
        self.dependencies[task_id].append(dependency_id)

    def find_order(self, task_ids) -> list:
        \"\"\"Finds a valid execution order using topological sort.\"\"\"
        visited = set()
        order = []
        for task_id in task_ids:
            if task_id not in visited:
                self._topo_sort(task_id, visited, order)
        return order

    def _topo_sort(self, task_id, visited, order):
        visited.add(task_id)
        for dep in self.dependencies.get(task_id, []):
            if dep not in visited:
                self._topo_sort(dep, visited, order)
        order.insert(0, task_id)
""",
    "task2": """from projects.task_scheduler_task2.graph import DependencyGraph

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
        \"\"\"Returns tasks that are PENDING and all dependencies are COMPLETED.\"\"\"
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

        runnable.sort(key=lambda t: t.priority)
        return runnable
""",
    "task3": """from projects.task_scheduler_task3.scheduler import TaskScheduler

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
        \"\"\"Finds downstream tasks that depend on the failed task and marks them as FAILED.\"\"\"
        for task_id, task in self.scheduler.tasks.items():
            if task.status == "PENDING":
                deps = self.scheduler.graph.dependencies.get(task_id, [])
                if failed_task_id in deps:
                    task.status = "PENDING"
                    self._propagate_failure(task_id)
"""
}

# ==============================================================================
# Correct Code Contents (To clean workspace at the end)
# ==============================================================================

CORRECT_CODES = {
    "task1": """class DependencyGraph:
    def __init__(self):
        self.dependencies = {}  # task_id -> list of dependencies

    def add_dependency(self, task_id, dependency_id):
        if task_id not in self.dependencies:
            self.dependencies[task_id] = []
        self.dependencies[task_id].append(dependency_id)

    def find_order(self, task_ids) -> list:
        \"\"\"Finds a valid execution order using topological sort.\"\"\"
        visited = set()
        order = []
        for task_id in task_ids:
            if task_id not in visited:
                self._topo_sort(task_id, visited, order)
        return order

    def _topo_sort(self, task_id, visited, order):
        visited.add(task_id)
        for dep in self.dependencies.get(task_id, []):
            if dep not in visited:
                self._topo_sort(dep, visited, order)
        order.append(task_id)
""",
    "task2": """from projects.task_scheduler_task2.graph import DependencyGraph

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
        \"\"\"Returns tasks that are PENDING and all dependencies are COMPLETED.\"\"\"
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
""",
    "task3": """from projects.task_scheduler_task3.scheduler import TaskScheduler

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
        \"\"\"Finds downstream tasks that depend on the failed task and marks them as FAILED.\"\"\"
        for task_id, task in self.scheduler.tasks.items():
            if task.status == "PENDING":
                deps = self.scheduler.graph.dependencies.get(task_id, [])
                if failed_task_id in deps:
                    task.status = "FAILED"
                    self._propagate_failure(task_id)
"""
}

FILE_PATHS = {
    "task1": "projects/task_scheduler_task1/graph.py",
    "task2": "projects/task_scheduler_task2/scheduler.py",
    "task3": "projects/task_scheduler_task3/executor.py"
}

TEST_PATHS = {
    "task1": "projects/task_scheduler_task1/test_scheduler.py",
    "task2": "projects/task_scheduler_task2/test_scheduler.py",
    "task3": "projects/task_scheduler_task3/test_scheduler.py"
}

PROMPTS = {
    "task1": (
        "Fix the dependency ordering issue in the task scheduler project. "
        "The test verifying that a task graph with dependencies correctly orders them is failing. "
        "Search the codebase to find where the task scheduler graph logic and tests are, "
        "fix ONLY the ordering logic in the graph file, and run the tests to confirm it passes."
    ),
    "task2": (
        "Fix the priority execution sorting issue in the task scheduler project. "
        "The test verifying that runnable tasks are ordered by priority (higher priority number first) is failing. "
        "Search the codebase to find where the task scheduler class and tests are, "
        "fix ONLY the priority sorting bug in the scheduler file, and run the tests to confirm it passes."
    ),
    "task3": (
        "Fix the failure status propagation issue in the task scheduler project. "
        "The test verifying that if a task fails, all its downstream dependent tasks are marked as FAILED is failing. "
        "Search the codebase to find where the task scheduler executor class and tests are, "
        "fix ONLY the status failure propagation logic in the executor file, and run the tests to confirm it passes."
    )
}

# ==============================================================================
# Helper functions
# ==============================================================================

def reset_file_to_buggy(task_id: str):
    """Write the exact buggy implementation to the target file."""
    path = FILE_PATHS[task_id]
    with open(path, "w", encoding="utf-8") as f:
        f.write(BUGGY_CODES[task_id])
    print(f"🧹 Reset {path} to buggy state.")

def restore_file_to_correct(task_id: str):
    """Write the clean correct implementation back to the target file."""
    path = FILE_PATHS[task_id]
    with open(path, "w", encoding="utf-8") as f:
        f.write(CORRECT_CODES[task_id])
    print(f"✨ Restored {path} to correct state.")

def verify_codebase_tests(task_id: str) -> bool:
    """Runs pytest using sys.executable on the target test file to verify success."""
    test_path = TEST_PATHS[task_id]
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", test_path],
            capture_output=True,
            text=True,
            timeout=10.0
        )
        return res.returncode == 0
    except Exception as e:
        print(f"⚠️ Pytest execution crashed: {e}")
        return False

# ==============================================================================
# Main Benchmark Runner
# ==============================================================================

async def run_benchmark():
    conditions = ["baseline", "rag"]
    tasks = ["task1", "task2", "task3"]
    runs_per_task = 3
    
    print("=" * 80)
    print("📋 STARTING WEEK 8 RAG VS BASELINE RETRIEVAL BENCHMARK")
    print("=" * 80)
    
    # Initialize SQLite table for the benchmark
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition TEXT,
            task_id TEXT,
            run_id TEXT,
            steps INTEGER,
            cost_usd REAL,
            retrieval_calls INTEGER,
            outcome TEXT,
            duration_s REAL
        )
    """)
    conn.commit()
    conn.close()

    for condition in conditions:
        for task_id in tasks:
            for run_num in range(1, runs_per_task + 1):
                print("\n" + "-" * 80)
                print(f"🎬 Condition: '{condition.upper()}' | Task: '{task_id.upper()}' | Iteration: {run_num}/3")
                print("-" * 80)
                
                # 1. Reset codebase to clean buggy state
                reset_file_to_buggy(task_id)
                
                # Verify it is indeed broken
                assert not verify_codebase_tests(task_id), f"Setup check failed: Task {task_id} is not failing tests."

                # 2. Record start timestamp matching AgentTracer format exactly
                # AgentTracer uses: datetime.now(timezone.utc).isoformat()
                start_timestamp_iso = datetime.now(timezone.utc).isoformat()
                start_time = time.perf_counter()

                # 3. Setup client and tools
                client = get_llm_client(provider=LLMProvider.CLAUDE)
                client.register_tool(WRITE_FILE_TOOL, write_file)
                client.register_tool(READ_FILE_TOOL, read_file)
                client.register_tool(RUN_PYTHON_TOOL, run_python)
                client.register_tool(RUN_TESTS_TOOL, run_tests)
                client.register_tool(EDIT_FILE_TOOL, edit_file)

                # Initialize optional RAG pipeline
                rag_pipeline = None
                if condition == "rag":
                    store = VectorStore()
                    openai_client = OpenAIClient()
                    rag_pipeline = RAGPipeline(store, openai_client)

                # Initialize ReAct Agent
                agent = ReActAgent(
                    client=client,
                    model=ClaudeModel.CLAUDE_SONNET_4_6,
                    max_steps=15,
                    rag_pipeline=rag_pipeline
                )

                # 4. Execute the agent using Reflexion loop
                prompt = PROMPTS[task_id]
                try:
                    result = await agent.run_with_reflection(prompt, max_attempts=3)
                except Exception as e:
                    print(f"❌ ReAct loop crashed with exception: {e}")
                    result = None

                # 5. Measure duration
                duration_s = time.perf_counter() - start_time
                print(f"⏱️ Finished. Time: {duration_s:.2f} seconds.")

                # 6. Verify success independently to catch false positives
                agent_claimed_success = result.success if result else False
                actual_success = verify_codebase_tests(task_id)
                
                if agent_claimed_success:
                    if actual_success:
                        outcome = "success"
                    else:
                        outcome = "false positive"
                else:
                    outcome = "failure"
                print(f"Outcome classification: Agent claimed success={agent_claimed_success}, Pytest verified={actual_success} -> Outcome='{outcome}'")

                # 7. Query DB for metrics across all attempts of this benchmark run
                total_steps = 0
                total_cost_usd = 0.0
                retrieval_calls = 0
                final_run_id = "unknown"

                try:
                    conn = sqlite3.connect(DEFAULT_DB_PATH)
                    cursor = conn.cursor()
                    
                    # Fetch all attempt runs matching task prompt started since our start timestamp
                    cursor.execute(
                        "SELECT run_id, total_cost_usd FROM agent_runs WHERE task = ? AND started_at >= ? ORDER BY started_at ASC",
                        (prompt, start_timestamp_iso)
                    )
                    runs = cursor.fetchall()
                    
                    if runs:
                        final_run_id = runs[-1][0] # take the last attempt's run_id as the primary run identifier
                        for r_id, r_cost in runs:
                            total_cost_usd += r_cost or 0.0
                            
                            # Count steps in agent_steps
                            cursor.execute("SELECT COUNT(*) FROM agent_steps WHERE run_id = ?", (r_id,))
                            total_steps += cursor.fetchone()[0]
                            
                            # Count retrieve_context tool calls
                            cursor.execute("SELECT COUNT(*) FROM agent_steps WHERE run_id = ? AND action = 'retrieve_context'", (r_id,))
                            retrieval_calls += cursor.fetchone()[0]
                    else:
                        # Fallback if DB write fails or missing
                        total_steps = len(result.steps) if (result and result.steps) else 0
                        total_cost_usd = 0.0
                        
                    # Persist to rag_benchmark_runs
                    cursor.execute("""
                        INSERT INTO rag_benchmark_runs (
                            condition, task_id, run_id, steps, cost_usd, retrieval_calls, outcome, duration_s
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        condition,
                        task_id,
                        final_run_id,
                        total_steps,
                        total_cost_usd,
                        retrieval_calls,
                        outcome,
                        duration_s
                    ))
                    conn.commit()
                    conn.close()
                    print(f"📊 Persisted run metrics: run_id={final_run_id}, steps={total_steps}, cost_usd=${total_cost_usd:.5f}, retrieval_calls={retrieval_calls}, duration_s={duration_s:.2f}")
                except Exception as e:
                    print(f"⚠️ Error logging run metrics to SQLite: {e}")

                # Clean target codebase files back to correct state to avoid leaving worktrees dirty
                restore_file_to_correct(task_id)

    # ==============================================================================
    # Post-Run Comparative Summary Table
    # ==============================================================================
    print("\n" + "=" * 90)
    print("📈 COMPARATIVE BENCHMARK RUNS METRICS TABLE")
    print("=" * 90)
    print(f"{'Condition':<10} | {'Task':<6} | {'Run':<3} | {'Outcome':<14} | {'Steps':<5} | {'Cost (USD)':<10} | {'RAG Calls':<9} | {'Duration (s)':<12}")
    print("-" * 90)
    
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.cursor()
        # Fetch all records
        cursor.execute("SELECT condition, task_id, steps, cost_usd, retrieval_calls, outcome, duration_s FROM rag_benchmark_runs ORDER BY id ASC")
        records = cursor.fetchall()
        conn.close()
        
        # Keep track of counts for printing
        counts = {}
        for condition, task_id, steps, cost, rag_calls, outcome, duration in records:
            key = f"{condition}_{task_id}"
            counts[key] = counts.get(key, 0) + 1
            run_num = counts[key]
            print(
                f"{condition:<10} | {task_id:<6} | {run_num:<3} | {outcome:<14} | {steps:<5} | "
                f"${cost:<9.5f} | {rag_calls:<9} | {duration:<12.1f}"
            )
    except Exception as e:
        print(f"Error printing comparative summary table: {e}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
