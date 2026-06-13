import os
import sqlite3
import asyncio
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

async def main():
    # 1. Initialize LLM Client for the agent (Claude)
    client = get_llm_client(provider=LLMProvider.CLAUDE)

    # 2. Register standard tools
    client.register_tool(WRITE_FILE_TOOL, write_file)
    client.register_tool(READ_FILE_TOOL, read_file)
    client.register_tool(RUN_PYTHON_TOOL, run_python)
    client.register_tool(RUN_TESTS_TOOL, run_tests)
    client.register_tool(EDIT_FILE_TOOL, edit_file)

    # 3. Initialize RAG components
    store = VectorStore()
    openai_client = OpenAIClient()
    rag_pipeline = RAGPipeline(store, openai_client)

    # 4. Initialize ReAct Agent with RAG Pipeline
    agent = ReActAgent(
        client=client,
        model=ClaudeModel.CLAUDE_SONNET_4_6,
        max_steps=15,
        rag_pipeline=rag_pipeline
    )

    # Task 2: Fix Priority Execution
    prompt = (
        "Fix the priority execution sorting issue in the task scheduler project. "
        "The test verifying that runnable tasks are ordered by priority (higher priority number first) is failing. "
        "Search the codebase to find where the task scheduler class and tests are, "
        "fix ONLY the priority sorting bug in the scheduler file, and run the tests to confirm it passes."
    )

    print("\n" + "=" * 80)
    print("🎬 Executing Smoke Test with RAG Pipeline on Task 2")
    print("=" * 80)
    
    result = await agent.run(prompt)
    
    print("\n📊 Run Result Summary:")
    print(f"  Success: {result.success}")
    print(f"  Total Steps: {result.total_steps}")
    print(f"  Final Answer: {result.answer}")
    
    # 5. Query the SQLite database trace logs to print the step-by-step trace
    print("\n" + "=" * 80)
    print("📈 Step-by-Step Agent Database Tracer Logs")
    print("=" * 80)
    
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.cursor()
        
        # Get the latest run ID
        cursor.execute("SELECT run_id, total_cost_usd, total_tokens, outcome FROM agent_runs ORDER BY finished_at DESC LIMIT 1")
        run = cursor.fetchone()
        if not run:
            print("No agent runs found in database.")
            return
            
        r_id, cost, tokens, outcome = run
        print(f"Run ID: {r_id} | Outcome: {outcome} | Total Cost: ${cost:.6f} | Total Tokens: {tokens}")
        print("-" * 80)
        
        cursor.execute("SELECT step_num, action, thought, observation FROM agent_steps WHERE run_id = ? ORDER BY step_num ASC", (r_id,))
        steps = cursor.fetchall()
        for s_num, action, thought, obs in steps:
            print(f"\n👉 Step {s_num}: Action: '{action}'")
            print(f"   Thought: {thought}")
            # print first 150 chars of observation
            short_obs = obs[:150].replace('\n', ' ') + "..." if len(obs) > 150 else obs.replace('\n', ' ')
            print(f"   Observation: {short_obs}")
            
        conn.close()
    except Exception as e:
        print(f"Error querying SQLite database: {e}")

if __name__ == "__main__":
    asyncio.run(main())
