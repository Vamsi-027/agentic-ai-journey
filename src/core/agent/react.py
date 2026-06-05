import re
import json
import asyncio
from typing import Optional, Union, Tuple
from src.core.llm import BaseLLMClient, ToolDefinition

REACT_SYSTEM_PROMPT = """You are an autonomous agent executing a task. You use a Thought, Action, Observation loop to solve tasks step-by-step.

For each turn, you MUST output exactly one Thought and one Action (or Final Answer) in the following format:

Thought: <your reasoning about what you need to do next>
Action: <tool_name> <a single-line JSON object containing the tool arguments>

After your Action, the user/system will execute the tool and provide you with an:
Observation: <result of the tool execution>

When you have gathered all the information and are ready to provide the final answer, output:
Thought: I have the final answer.
Final Answer: <your final answer to the user>

Rules:
1. You MUST only output ONE Thought and ONE Action per turn. Never output multiple actions.
2. The Action block arguments MUST be a single-line valid JSON object. E.g. Action: write_file {"path": "test.txt", "content": "hello"}
3. If you do not need to call any more tools, you MUST output 'Final Answer: <your final response>' to terminate the loop.

Available Tools:
- write_file: Write content to a file. Arguments: {"path": "string", "content": "string"}
- read_file: Read and return complete content of a file. Arguments: {"path": "string"}
- run_python: Run Python code block in a subprocess with a timeout and get stdout/stderr. Arguments: {"code": "string"}
- search_web: Search the web for a query (currently a stub). Arguments: {"query": "string"}
"""

def parse_react_action(text: str) -> Tuple[str, str, dict]:
    """Parses LLM output text for Action or Final Answer blocks.
    
    Returns:
        tuple: (status, tool_name_or_answer, arguments_dict)
        where status can be:
        - "action": tool_name is populated, arguments contains parsed JSON args
        - "final": tool_name_or_answer contains the final answer text
        - "error": tool_name_or_answer contains error description, arguments has raw arguments string
        - "none": no parseable block was found
    """
    # Look for Action: tool_name {"args": ...}
    action_match = re.search(r"Action:\s*(\w+)\s*(\{.*\})", text)
    if action_match:
        tool_name = action_match.group(1).strip()
        args_str = action_match.group(2).strip()
        try:
            args = json.loads(args_str)
            return "action", tool_name, args
        except json.JSONDecodeError:
            return "error", "Failed to parse Action arguments as valid JSON", {"raw_args": args_str}

    # Look for Final Answer: ...
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return "final", final_match.group(1).strip(), {}

    return "none", "", {}


class ReActAgent:
    """An autonomous agent implementing a pure Python text-based ReAct loop."""

    def __init__(self, client: BaseLLMClient, model: Optional[str] = None, system_prompt: str = REACT_SYSTEM_PROMPT, max_steps: int = 10):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def run(self, task: str) -> str:
        """Runs the ReAct loop to complete the given task, logging each intermediate step."""
        messages = [{"role": "user", "content": f"Task: {task}"}]
        
        for step in range(self.max_steps):
            print(f"\n🚀 === [AGENT STEP {step + 1}] ===")
            
            # 1. Generate next step
            response = await self.client.generate(
                prompt=messages,
                system_prompt=self.system_prompt,
                model=self.model,
                temperature=0.0
            )
            llm_output = response.text
            print(f"\n[Agent Thought & Action]:\n{llm_output}\n")
            
            # Append agent response to history
            messages.append({"role": "assistant", "content": llm_output})
            
            # 2. Parse the step
            status, value, args = parse_react_action(llm_output)
            
            if status == "final":
                print(f"✅ Final Answer Reached: {value}")
                return value
                
            elif status == "error":
                observation = f"Error: Action arguments must be a single-line valid JSON object. Got parsing error for: {args['raw_args']}"
                print(f"❌ {observation}")
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue
                
            elif status == "none":
                # Fallback check for raw "Final Answer" string in case regex missed formatting
                if "Final Answer:" in llm_output:
                    answer = llm_output.split("Final Answer:")[-1].strip()
                    print(f"✅ Final Answer Reached (fallback): {answer}")
                    return answer
                
                observation = "Error: Invalid output format. You must output exactly: 'Thought: <reasoning>\\nAction: <tool_name> <arguments_json>' or 'Final Answer: <answer>'."
                print(f"❌ {observation}")
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue
            
            # 3. Dispatch the tool call
            tool_name = value
            print(f"🔧 Dispatching tool: '{tool_name}' with args: {args}")
            try:
                observation = await self.client.dispatch(tool_name, args)
            except KeyError:
                observation = f"Error: Tool '{tool_name}' is not registered. Registered tools: write_file, read_file, run_python, search_web"
            except Exception as e:
                observation = f"Error executing tool: {str(e)}"
                
            print(f"📥 Observation: {observation}")
            
            # Feed tool outcome back as user message
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            
        print("⚠️ Maximum steps reached without completing the task.")
        return "Failed to complete task within maximum steps limit."
