import os
import asyncio
from src.core.llm import (
    get_llm_client,
    LLMProvider,
    ClaudeModel,
    WRITE_FILE_TOOL,
    READ_FILE_TOOL,
    RUN_PYTHON_TOOL,
    SEARCH_WEB_TOOL,
    write_file,
    read_file,
    run_python,
    search_web
)
from src.core.agent.react import ReActAgent

async def run_tasks():
    # 1. Initialize Claude Client
    client = get_llm_client(provider=LLMProvider.CLAUDE)
    
    # 2. Register tools
    client.register_tool(WRITE_FILE_TOOL, write_file)
    client.register_tool(READ_FILE_TOOL, read_file)
    client.register_tool(RUN_PYTHON_TOOL, run_python)
    client.register_tool(SEARCH_WEB_TOOL, search_web)
    
    # 3. Setup files for Task 1
    x_path = "X.txt"
    y_path = "Y.txt"
    search_result_path = "search_result.txt"
    
    # Clean up previous runs
    for p in [x_path, y_path, search_result_path]:
        if os.path.exists(p):
            os.remove(p)
            
    # Write source text for Task 1
    source_content = (
        "Quantum computing leverages superposition and entanglement to perform calculations "
        "that would take classical systems millions of years. Superposition allows qubits to exist "
        "in multiple states simultaneously, while entanglement links them instantly across any distance. "
        "However, maintaining qubit stability (preventing decoherence) is the primary engineering barrier."
    )
    with open(x_path, "w", encoding="utf-8") as f:
        f.write(source_content)
    print(f"Pre-populated {x_path} with: '{source_content}'")
    
    # Initialize ReAct Agent with the supported model
    agent = ReActAgent(client=client, model=ClaudeModel.CLAUDE_SONNET_4_6, max_steps=8)
    
    # ==========================================================================
    # TASK 1: Read X.txt, summarize it, write summary to Y.txt
    # ==========================================================================
    print("\n" + "="*80)
    print("📋 TASK 1: Read X.txt, summarize it, write summary to Y.txt")
    print("="*80)
    task1_prompt = f"Read the file '{x_path}', summarize its key points in one sentence, and write that summary to the file '{y_path}'."
    task1_result = await agent.run(task1_prompt)
    
    # Check if Y.txt exists and output its content
    if os.path.exists(y_path):
        with open(y_path, "r", encoding="utf-8") as f:
            summary_content = f.read()
        print(f"\n📝 Verification: Content of {y_path} is: '{summary_content}'")
    else:
        print(f"\n⚠️ Verification: {y_path} was not created!")
        
    # ==========================================================================
    # TASK 2: Calculate compound interest
    # ==========================================================================
    print("\n" + "="*80)
    print("📋 TASK 2: Calculate compound interest")
    print("="*80)
    task2_prompt = (
        "Calculate the compound interest for a principal of $10,000, an annual interest rate "
        "of 5% compounded monthly, for 10 years. You MUST use the run_python tool to execute Python code "
        "to calculate the final amount (A) and total interest earned (I). Print the final values in your Final Answer."
    )
    task2_result = await agent.run(task2_prompt)
    
    # ==========================================================================
    # TASK 3: A 3-step chain of your choice (Search -> Write -> Read & Print Uppercase)
    # ==========================================================================
    print("\n" + "="*80)
    print("📋 TASK 3: 3-step Chain (Search -> Write -> Subprocess Read & Uppercase)")
    print("="*80)
    task3_prompt = (
        f"1. Search the web for 'Python async events'.\n"
        f"2. Write the search results directly to '{search_result_path}'.\n"
        f"3. Run a Python script via the run_python tool to read '{search_result_path}' and print its content in UPPERCASE."
    )
    task3_result = await agent.run(task3_prompt)


if __name__ == "__main__":
    asyncio.run(run_tasks())
