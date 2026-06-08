import os
import sqlite3
import asyncio
from src.core.llm import get_llm_client, LLMProvider, ClaudeModel
from src.core.llm.tools import (
    WRITE_FILE_TOOL, write_file,
    READ_FILE_TOOL, read_file,
    RUN_PYTHON_TOOL, run_python,
    RUN_TESTS_TOOL, run_tests,
    SEARCH_WEB_TOOL, search_web,
    LIST_DIRECTORY_TOOL, list_directory,
    SEARCH_IN_FILES_TOOL, search_in_files,
    EDIT_FILE_TOOL, edit_file
)
from src.core.agent.react import ReActAgent
from src.core.database import DEFAULT_DB_PATH

async def run_integration_tests():
    # Initialize LLM Client
    client = get_llm_client(provider=LLMProvider.CLAUDE)

    # Register all tools
    client.register_tool(WRITE_FILE_TOOL, write_file)
    client.register_tool(READ_FILE_TOOL, read_file)
    client.register_tool(RUN_PYTHON_TOOL, run_python)
    client.register_tool(RUN_TESTS_TOOL, run_tests)
    client.register_tool(SEARCH_WEB_TOOL, search_web)
    client.register_tool(LIST_DIRECTORY_TOOL, list_directory)
    client.register_tool(SEARCH_IN_FILES_TOOL, search_in_files)
    client.register_tool(EDIT_FILE_TOOL, edit_file)

    # We will use Claude 3.5 Sonnet to execute these runs
    agent = ReActAgent(client=client, model=ClaudeModel.CLAUDE_SONNET_4_6, max_steps=15)

    prompts = [
        # Task 1: Fix graph.py topo_sort bug
        (
            "Task 1: Fix Dependency Ordering",
            "The test test_dependency_ordering in projects/task_scheduler_task1/test_scheduler.py is failing. "
            "It verifies that a task graph with dependencies correctly orders them (dependencies execute first). "
            "Fix ONLY the ordering logic in projects/task_scheduler_task1/graph.py. "
            "Do not modify any other files or tests. "
            "Verify by running pytest: projects/task_scheduler_task1/test_scheduler.py::test_dependency_ordering"
        ),
        # Task 2: Fix scheduler.py priority sort bug
        (
            "Task 2: Fix Priority Execution",
            "The test test_priority_execution in projects/task_scheduler_task2/test_scheduler.py is failing. "
            "It verifies that runnable tasks are ordered by priority (higher priority number first). "
            "Fix ONLY the priority sorting logic in projects/task_scheduler_task2/scheduler.py. "
            "Do not modify any other files or tests. "
            "Verify by running pytest: projects/task_scheduler_task2/test_scheduler.py::test_priority_execution"
        ),
        # Task 3: Fix executor.py status failure propagation bug
        (
            "Task 3: Fix Failure Propagation",
            "The test test_dependency_failure_propagation in projects/task_scheduler_task3/test_scheduler.py is failing. "
            "It verifies that if a task fails, all its downstream dependent tasks are marked as FAILED. "
            "Fix ONLY the failure propagation status logic in projects/task_scheduler_task3/executor.py. "
            "Do not modify any other files or tests. "
            "Verify by running pytest: projects/task_scheduler_task3/test_scheduler.py::test_dependency_failure_propagation"
        )
    ]

    results = []

    for name, prompt in prompts:
        print("\n" + "=" * 80)
        print(f"🎬 Executing Integration {name}")
        print("=" * 80)
        
        # We run the agent with reflexion loop
        result = await agent.run_with_reflection(prompt, max_attempts=3)
        
        print(f"\n📊 {name} Result Summary:")
        print(f"  Success: {result.success}")
        print(f"  Total Steps: {result.total_steps}")
        print(f"  Final Answer: {result.answer}")
        
        results.append({
            "name": name,
            "success": result.success,
            "total_steps": result.total_steps,
            "answer": result.answer
        })

    # Query the SQLite database trace logs to print empirical summary
    print("\n" + "=" * 80)
    print("📈 Empirical Database Tracer Summary")
    print("=" * 80)
    
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.cursor()
        
        # Get the runs ordered by finished_at descending (last 10 runs to capture retries/reflexion runs)
        cursor.execute("""
            SELECT run_id, task, total_cost_usd, total_tokens, outcome, finished_at
            FROM agent_runs
            ORDER BY finished_at DESC
            LIMIT 12
        """)
        runs = cursor.fetchall()
        
        print(f"{'Run ID':<40} | {'Outcome':<8} | {'Cost (USD)':<10} | {'Tokens':<8} | {'Task'}")
        print("-" * 120)
        for r_id, task, cost, tokens, outcome, finished in runs:
            # We crop the task string to look neat
            short_task = task[:50] + "..." if len(task) > 50 else task
            print(f"{r_id:<40} | {outcome:<8} | ${cost:<9.6f} | {tokens:<8} | {short_task}")
            
            # Print steps details for this run
            cursor.execute("""
                SELECT step_num, action, latency_ms, tokens_used
                FROM agent_steps
                WHERE run_id = ?
                ORDER BY step_num ASC
            """, (r_id,))
            steps = cursor.fetchall()
            for s_num, action, latency, step_tokens in steps:
                print(f"   Step {s_num}: {action[:30]:<30} | Latency: {latency:<6} ms | Tokens: {step_tokens}")
                
        conn.close()
    except Exception as e:
        print(f"Error querying SQLite database: {e}")

if __name__ == "__main__":
    asyncio.run(run_integration_tests())
