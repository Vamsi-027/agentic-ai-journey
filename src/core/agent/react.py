import re
import json
import asyncio
from typing import Optional, Union, Tuple
from src.core.llm import BaseLLMClient, ToolDefinition
from src.core.retrieval.rag_pipeline import RAGPipeline

from dataclasses import dataclass

REACT_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent executing a task. You use a Thought, Action, Observation loop to solve tasks step-by-step.

For each turn, you MUST output exactly one Thought and one Action (or Final Answer) in the following format:

Thought: <your reasoning about what you need to do next>
Action: <tool_name> <arguments_json>

After your Action, the user/system will execute the tool and provide you with an:
Observation: <result of the tool execution>

When you have gathered all the information and are ready to provide the final answer, output:
Thought: I have the final answer.
Final Answer: <your final answer to the user>

Rules:
1. You MUST only output ONE Thought and ONE Action per turn. Never output multiple actions.
2. The Action block arguments MUST be a valid JSON object. E.g. Action: write_file {"path": "test.txt", "content": "hello"}
3. If you do not need to call any more tools, you MUST output 'Final Answer: <your final response>' to terminate the loop.
4. When pytest returns Return Code: 0 for the target test, your very next output must be Final Answer. Do not write an intermediate thought or summary.
5. You have access to a retrieve_context tool that searches the codebase for relevant code. Use it before reading files when you need to locate where something is implemented.

Available Tools:
{{TOOLS}}
"""

REACT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT_TEMPLATE


@dataclass
class AgentResult:
    answer: str
    steps: list[
        dict
    ]  # each: {"step": int, "thought": str, "action": str, "observation": str}
    success: bool
    total_steps: int


def _extract_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    in_string_char = None
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2  # skip escaped character
                continue
            elif ch == in_string_char:
                in_string = False
                in_string_char = None
        else:
            if ch in ('"', "'"):
                in_string = True
                in_string_char = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1
    return None


def _log_agent_step(
    tracer,
    response,
    step_start_time: float,
    thought: str,
    action: str,
    action_input: str,
    observation: str,
) -> Tuple[int, float]:
    import time
    latency_ms = int((time.time() - step_start_time) * 1000)
    tokens = (response.input_tokens + response.output_tokens) if response else 0
    cost = getattr(response, "cost_usd", 0.0) or 0.0

    tracer.log_step(
        thought=thought,
        action=action,
        action_input=action_input,
        observation=observation,
        latency_ms=latency_ms,
        tokens_used=tokens,
    )
    return tokens, cost


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
    # Look for Action: tool_name
    action_match = re.search(r"Action:\s*(\w+)\s*", text)
    if action_match:
        tool_name = action_match.group(1).strip()
        args_str = _extract_json(text[action_match.end():])
        if args_str:
            try:
                args = json.loads(args_str)
                return "action", tool_name, args
            except json.JSONDecodeError:
                return (
                    "error",
                    "Failed to parse Action arguments as valid JSON",
                    {"raw_args": args_str},
                )
        else:
            return (
                "error",
                "Action arguments JSON block not found or unbalanced",
                {"raw_args": text[action_match.end():].strip()},
            )

    # Look for Final Answer: ...
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return "final", final_match.group(1).strip(), {}

    return "none", "", {}


class ReActAgent:
    """An autonomous agent implementing a pure Python text-based ReAct(Reason + Act) loop."""

    def __init__(
        self,
        client: BaseLLMClient,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        rag_pipeline: Optional[RAGPipeline] = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT_TEMPLATE
        self.max_steps = max_steps
        self.rag_pipeline = rag_pipeline
        
        # If rag_pipeline is provided, register retrieve_context tool
        if self.rag_pipeline:
            from src.core.llm import RETRIEVE_CONTEXT_TOOL, retrieve_context
            self.client.register_tool(RETRIEVE_CONTEXT_TOOL, retrieve_context)

    def _build_system_prompt(self) -> str:
        parts = []
        for name, (tool_def, _) in self.client.registry.items():
            props = tool_def.input_schema.get("properties", {})
            required = set(tool_def.input_schema.get("required", []))
            args_desc = ", ".join(
                f"{k} ({'required' if k in required else 'optional'}): {v.get('description', v.get('type', ''))}"
                for k, v in props.items()
            )
            parts.append(f"- {name}: {tool_def.description}\n  Args: {args_desc}")
        tools_section = "\n".join(parts)
        if "{{TOOLS}}" in self.system_prompt:
            return self.system_prompt.replace("{{TOOLS}}", tools_section)
        return self.system_prompt

    async def run(self, task: str) -> AgentResult:
        """Runs the ReAct loop to complete the given task, logging each intermediate step to SQLite tracing."""
        from src.core.database import AgentTracer
        import time

        messages = [{"role": "user", "content": f"Task: {task}"}]
        steps = []
        sys_prompt = self._build_system_prompt()
        seen_actions: set[str] = set()

        total_cost_usd = 0.0
        total_tokens = 0
        model_name = (
            self.model.value
            if hasattr(self.model, "value")
            else str(self.model or "unknown-model")
        )

        with AgentTracer(task=task, model=model_name) as tracer:
            if self.rag_pipeline:
                import os
                import src.core.llm.tools
                from src.core.config import settings
                workspace_path = settings.WORKSPACE_ROOT
                print(f"Indexing workspace root: {workspace_path}...")
                start_time_idx = time.perf_counter()
                chunk_count, embedding_cost = await self.rag_pipeline.index_directory(
                    workspace_path, extensions=[".py"]
                )
                indexing_latency_ms = int((time.perf_counter() - start_time_idx) * 1000)
                print(f"Indexed {chunk_count} chunks, cost: ${embedding_cost:.6f}")
                
                # Persist the freshly built index to disk so the retrieval tool can load it
                index_base = os.path.join(workspace_path, "data", "codebase_index")
                self.rag_pipeline.vector_store.save(index_base)
                # Reset retrieve_context module-level cache to force reloading the new index
                src.core.llm.tools._RAG_PIPELINE = None
                print(f"Persisted freshly built codebase index to: {index_base}")
                
                # Log indexing run to the tracer
                total_cost_usd += embedding_cost
                tracer.total_cost_usd = total_cost_usd
                tracer.log_step(
                    thought="System setup: Indexing workspace directories for context retrieval.",
                    action="index_directory",
                    action_input=json.dumps({"workspace_path": workspace_path, "extensions": [".py"]}),
                    observation=f"Indexed {chunk_count} chunks. Embedding cost: ${embedding_cost:.6f}",
                    latency_ms=indexing_latency_ms,
                    tokens_used=0
                )


            for step in range(self.max_steps):
                step_start_time = time.time()
                print(f"\n🚀 === [AGENT STEP {step + 1}] ===")

                # 1. Generate next step passing stop sequence to prevent hallucinating observation
                response = await self.client.chat(
                    messages=messages,
                    system_prompt=sys_prompt,
                    model=self.model,
                    temperature=0.0,
                    stop=["Observation:"]
                )
                llm_output = response.text
                print(f"\n[Agent Thought & Action]:\n{llm_output}\n")

                # Append agent response to history
                messages.append({"role": "assistant", "content": llm_output})

                # Extract Thought
                thought_match = re.search(
                    r"Thought:\s*(.*?)(?=\s*(Action:|Final Answer:|$))",
                    llm_output,
                    re.DOTALL,
                )
                thought_text = (
                    thought_match.group(1).strip()
                    if thought_match
                    else llm_output.strip()
                )

                # 2. Parse the step
                status, value, args = parse_react_action(llm_output)

                if status == "final":
                    print(f"✅ Final Answer Reached: {value}")
                    steps.append(
                        {
                            "step": step + 1,
                            "thought": thought_text,
                            "action": f"Final Answer: {value}",
                            "observation": "",
                        }
                    )

                    step_tokens, step_cost = _log_agent_step(
                        tracer, response, step_start_time,
                        thought=thought_text,
                        action="Final Answer",
                        action_input=json.dumps({"answer": value}),
                        observation=""
                    )
                    total_tokens += step_tokens
                    total_cost_usd += step_cost
                    tracer.total_cost_usd = total_cost_usd
                    tracer.total_tokens = total_tokens
                    tracer.outcome = "success"

                    return AgentResult(
                        answer=value, steps=steps, success=True, total_steps=len(steps)
                    )

                elif status == "error":
                    observation = f"Error: Action arguments must be a valid JSON object. Got parsing error for: {args['raw_args']}"
                    print(f"❌ {observation}")
                    steps.append(
                        {
                            "step": step + 1,
                            "thought": thought_text,
                            "action": f"Error Action: {value}",
                            "observation": observation,
                        }
                    )
                    messages.append(
                        {"role": "user", "content": f"Observation: {observation}"}
                    )

                    step_tokens, step_cost = _log_agent_step(
                        tracer, response, step_start_time,
                        thought=thought_text,
                        action=f"Error Action: {value}",
                        action_input=json.dumps(args),
                        observation=observation
                    )
                    total_tokens += step_tokens
                    total_cost_usd += step_cost
                    tracer.total_cost_usd = total_cost_usd
                    tracer.total_tokens = total_tokens
                    continue

                elif status == "none":
                    # Fallback check for raw "Final Answer" string in case regex missed formatting
                    if "Final Answer:" in llm_output:
                        answer = llm_output.split("Final Answer:")[-1].strip()
                        print(f"✅ Final Answer Reached (fallback): {answer}")
                        steps.append(
                            {
                                "step": step + 1,
                                "thought": thought_text,
                                "action": f"Final Answer: {answer}",
                                "observation": "",
                            }
                        )

                        step_tokens, step_cost = _log_agent_step(
                            tracer, response, step_start_time,
                            thought=thought_text,
                            action="Final Answer",
                            action_input=json.dumps({"answer": answer}),
                            observation=""
                        )
                        total_tokens += step_tokens
                        total_cost_usd += step_cost
                        tracer.total_cost_usd = total_cost_usd
                        tracer.total_tokens = total_tokens
                        tracer.outcome = "success"

                        return AgentResult(
                            answer=answer,
                            steps=steps,
                            success=True,
                            total_steps=len(steps),
                        )

                    observation = "Error: Invalid output format. You must output exactly: 'Thought: <reasoning>\\nAction: <tool_name> <arguments_json>' or 'Final Answer: <answer>'."
                    print(f"❌ {observation}")
                    steps.append(
                        {
                            "step": step + 1,
                            "thought": thought_text,
                            "action": "None",
                            "observation": observation,
                        }
                    )
                    messages.append(
                        {"role": "user", "content": f"Observation: {observation}"}
                    )

                    step_tokens, step_cost = _log_agent_step(
                        tracer, response, step_start_time,
                        thought=thought_text,
                        action="None",
                        action_input=json.dumps({}),
                        observation=observation
                    )
                    total_tokens += step_tokens
                    total_cost_usd += step_cost
                    tracer.total_cost_usd = total_cost_usd
                    tracer.total_tokens = total_tokens
                    continue

                # 3. Dispatch the tool call
                tool_name = value
                action_text = f"{tool_name} {json.dumps(args)}"
                print(f"🔧 Dispatching tool: '{tool_name}' with args: {args}")

                # Loop detection: check if this action with these args has been run before
                action_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if action_key in seen_actions:
                    observation = (
                        f"Error: You already called '{tool_name}' with these exact arguments. "
                        "You already know this result. Review your observations and try a different approach."
                    )
                    print(f"🔁 Loop detected: {observation}")
                else:
                    seen_actions.add(action_key)
                    try:
                        observation = await self.client.dispatch(tool_name, args)
                    except KeyError:
                        registered = list(self.client.registry.keys())
                        observation = f"Error: Tool '{tool_name}' is not registered. Available tools: {registered}"
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"

                print(f"📥 Observation: {observation}")

                steps.append(
                    {
                        "step": step + 1,
                        "thought": thought_text,
                        "action": action_text,
                        "observation": observation,
                    }
                )

                # Feed tool outcome back as user message
                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )

                step_tokens, step_cost = _log_agent_step(
                    tracer, response, step_start_time,
                    thought=thought_text,
                    action=tool_name,
                    action_input=json.dumps(args),
                    observation=observation
                )
                total_tokens += step_tokens
                total_cost_usd += step_cost
                tracer.total_cost_usd = total_cost_usd
                tracer.total_tokens = total_tokens

            print("⚠️ Maximum steps reached without completing the task.")
            tracer.outcome = "failure"
            tracer.error_msg = "Maximum steps reached without completing the task."
            return AgentResult(
                answer="Failed to complete task within maximum steps limit.",
                steps=steps,
                success=False,
                total_steps=len(steps),
            )

    async def run_with_reflection(
        self, task: str, max_attempts: int = 3
    ) -> AgentResult:
        """Executes the task using a Reflexion loop (attempt -> evaluate -> reflect -> retry).
        Capped at max_attempts (default 3).
        """
        critique = None
        attempt = 1

        while attempt <= max_attempts:
            print(f"\n🔮 === [REFLEXION ATTEMPT {attempt}/{max_attempts}] ===")

            # Prepend the critique to the system prompt if we have one
            sys_prompt = self.system_prompt
            if critique:
                sys_prompt = (
                    f"### PREVIOUS ATTEMPT SELF-CRITIQUE & LESSONS LEARNED\n"
                    f"{critique}\n"
                    f"Use the self-critique above to improve your reasoning, correct your mistakes, and succeed in this attempt.\n"
                    f"###\n\n"
                ) + self.system_prompt

            # Temporarily override system prompt
            original_sys_prompt = self.system_prompt
            self.system_prompt = sys_prompt

            try:
                result = await self.run(task)
            except Exception as e:
                # Wrap loop exception as a failed AgentResult to allow reflection/retry
                print(f"❌ Attempt {attempt} crashed: {e}")
                result = AgentResult(
                    answer=f"Error executing agent run: {str(e)}",
                    steps=[],
                    success=False,
                    total_steps=0,
                )
            finally:
                self.system_prompt = original_sys_prompt

            # Evaluate success
            model_name = (
                self.model.value
                if hasattr(self.model, "value")
                else str(self.model or "unknown-model")
            )
            success, reason = await evaluate_success(
                self.client, task, result.answer, steps=result.steps, model=model_name
            )
            print(
                f"📊 Attempt {attempt} Evaluation: Success={success}, Reason={reason}"
            )

            # Update the result's success flag based on our evaluator
            result.success = success

            if success or attempt == max_attempts:
                return result

            print("🤔 Attempt failed. Generating self-critique...")
            critique = await generate_reflection(
                self.client, task, result.steps, model=model_name
            )
            print(f"📝 Critique generated:\n{critique}\n")

            attempt += 1


async def evaluate_success(
    client: BaseLLMClient,
    task: str,
    final_answer: str,
    steps: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> Tuple[bool, str]:
    """Evaluates task completion using rule-based checks and an LLM-as-judge call.
    Returns (success, reason).
    """
    # 1. Rule-based checks for tests and errors in the last tool observation
    last_observation = ""
    if steps:
        for step in reversed(steps):
            obs = step.get("observation", "")
            if obs:
                last_observation = obs
                break

    lower_obs = last_observation.lower()
    if "return code: 1" in lower_obs or (
        "fail" in lower_obs and ("pytest" in lower_obs or "test" in lower_obs)
    ):
        if "passed" not in lower_obs:
            return (
                False,
                "Rule-based check: Detected test failure or non-zero exit code in output.",
            )

    # 2. LLM-as-judge verification
    prompt = (
        "You are an objective AI judge evaluating if an autonomous coding agent succeeded at its task.\n\n"
        f"Task Description:\n{task}\n\n"
        f"Agent's Final Answer:\n{final_answer}\n\n"
        "Assess if the agent completed the task successfully based on its final answer and logs.\n"
        "Output your evaluation strictly in the following JSON format (no other text, no markdown blocks):\n"
        '{"success": true, "reason": "Reason for success"}\n'
        "or\n"
        '{"success": false, "reason": "Reason for failure"}'
    )

    try:
        response = await client.chat(
            messages=[{"role": "user", "content": prompt}], temperature=0.0, model=model
        )
        text = response.text.strip()
        json_str = _extract_json(text)
        if json_str:
            data = json.loads(json_str)
            return bool(data.get("success", False)), str(
                data.get("reason", "No reason provided.")
            )
        else:
            return False, f"LLM judge response did not contain a valid JSON block: {text}"
    except Exception as e:
        # Fallback to False if LLM judge fails to be safe and trigger retry
        return False, f"LLM judge failed; assumed failure to be safe. Error: {str(e)}"


async def generate_reflection(
    client: BaseLLMClient, task: str, steps: list[dict], model: Optional[str] = None
) -> str:
    """Calls the LLM to reflect on a failed step-by-step trace and generate a self-critique."""
    if not steps:
        return (
            "The agent failed before completing any steps. "
            "On the next attempt, start with list_directory to orient yourself, "
            "then read relevant source files before attempting any edits."
        )

    # Find the last non-empty tool observation
    last_observation = ""
    for s in reversed(steps):
        obs = s.get("observation", "")
        if obs:
            last_observation = obs
            break

    lower_obs = last_observation.lower()
    if "return code: 0" in lower_obs or "1 passed" in lower_obs:
        return (
            "The tests are already passing (Return Code: 0 / passed). "
            "Your previous attempt successfully completed the task and passed the tests, "
            "but did not output the Final Answer. "
            "In this attempt, immediately output Final Answer to conclude the task."
        )

    trace_lines = []
    for s in steps:
        trace_lines.append(
            f"Step {s['step']}:\n"
            f"  Thought: {s['thought']}\n"
            f"  Action: {s['action']}\n"
            f"  Observation: {s['observation']}\n"
        )
    trace_str = "\n".join(trace_lines)

    prompt = (
        "You attempted this task and failed. Here is your step-by-step execution trace:\n\n"
        f"Task:\n{task}\n\n"
        f"Step-by-step Trace:\n{trace_str}\n\n"
        "What specifically went wrong? What will you do differently in your next attempt?\n"
        "Generate a concise self-critique and action plan."
    )

    try:
        response = await client.chat(
            messages=[{"role": "user", "content": prompt}], temperature=0.0, model=model
        )
        return response.text.strip()
    except Exception as e:
        return f"Self-critique generation failed: {str(e)}"
