# Tool Registry & Multi-Turn Loop Design Decisions

This document details the architectural decisions made when extending the unified asynchronous LLM client with tool registry support and multi-turn execution loops.

## 1. Why a Registry Pattern?
We chose a **Registry Pattern** (storing a dictionary mapping `tool_name -> (ToolDefinition, Callable)`) at the client instance level for several key reasons:
- **Clean Separation of Concerns:** The model clients (`ClaudeClient`, `OpenAIClient`) remain oblivious to the specific implementations of developer tools. They simply load the tool schema definitions for api parameter validation, while execution is handled by dispatch lookup.
- **Provider Agnosticism:** By registering the tools in the registry on the abstract `BaseLLMClient`, the same registry works seamlessly across both Anthropic and OpenAI. The developer writes a tool function once and can run the multi-turn loop with any model or provider.
- **Dynamic Capabilities:** Subsystems or agents can dynamically register new tools during execution based on task state, context, or credentials, instead of relying on hardcoded static utility lists.

## 2. Loop Termination Conditions
The automated loop inside `run_with_tools` executes a series of turns to prevent infinite calls. It terminates under two main conditions:
1. **Implicit Termination (Standard Flow):** If the model returns a response where the `tool_calls` list is empty, it means the model has finished gathering tool facts and is presenting the final text response.
2. **Safety Turn Limit (Constraint Flow):** To prevent runaway infinite tool loops (which can happen if a model hallucinates arguments, fails repeatedly, or gets stuck in a loop), we enforce a `max_turns` limit (default `10`). Once reached, the loop terminates immediately and returns the last accumulated response.

In all cases, the final `LLMResponse` contains the aggregated token usage counts (`input_tokens` and `output_tokens`) and accumulated `cost_usd` accrued across all model turns.

## 3. Tool Error Handling
Tool execution errors are partitioned into two categories:
- **Registry / Contract Errors:** If the LLM generates a tool request for a name not registered in our dispatcher, `dispatch` raises a clean `KeyError`. The loop does not attempt to swallow this, because it indicates a configuration mismatch between the model's instructions and our registry.
- **Runtime Tool Execution Errors:** If a registered tool function throws an exception during execution (e.g. `read_file` fails due to `FileNotFoundError`, or python code syntax error in `run_python`), the loop catches it gracefully. It formats the error as a string (`"Error executing tool: <traceback>"`), flags `is_error=True` (for Claude content blocks), and passes it back to the model as the tool result. This allows the model to realize the error, recover, or report it cleanly to the user.

---

## 4. What I'd Do Differently (Retrospective)
If scaling this system to support production agentic flows, the following improvements would be prioritized:

1. **Parallel Execution of Tool Calls:** Currently, if the model returns multiple parallel tool calls in a single turn, we execute them sequentially using `for tool_call in response.tool_calls`. We should run them concurrently using `asyncio.gather(*tasks)` to minimize overall request latency when running multiple operations (e.g., calling multiple search queries or writing multiple files).
2. **State and Working Directory Sandboxing:** The `write_file`, `read_file`, and `run_python` tools execute relative to the host's current working directory. For security and stability, we should implement path-confinement (sandboxing paths inside a `.sandbox/` directory) and run python code inside Docker containers or virtual sandboxes (e.g., WASM/Deno) to prevent arbitrary host commands.
3. **Session-Level Logging Middleware:** Implementing hook callbacks before and after tool execution so that an external UI or telemetry manager (like Langfuse or OpenTelemetry) can track agent trace hierarchies without polluting the client code.
