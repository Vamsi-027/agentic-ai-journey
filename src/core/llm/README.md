# LLM Provider Client Module

## Problem Statement
Developing LLM-powered applications using vendor-specific SDKs directly inside core business logic introduces tight coupling, increases refactoring friction, and leaks implementation details across the codebase. Each provider—whether Anthropic or OpenAI—defines custom structures for input configurations, completion payloads, stream events, and function/tool declarations. This fragmentation obscures telemetry metrics like token counts, processing latency, and operational cost, making it difficult to swap models, run cross-provider benchmarks, or audit token usage without rewriting large portions of the system.

## Design Philosophy
This module implements a provider-agnostic, async-first adapter architecture that abstracts LLM execution behind a unified contract (`BaseLLMClient`). Standardized inputs map cleanly onto a normalized `LLMResponse` dataclass that automatically intercepts, calculates, and exposes token tracking and dollar pricing at runtime. By forcing async APIs, the engine promotes high-concurrency stream consumption and parallel orchestrations, while a declarative backoff-retry framework isolates network anomalies from the caller. 

## Quickstart
Initialize a client (automatically pulling configuration from environment settings or accepting explicit keys) and call the standard generator.

```python
import asyncio
from src.core.llm import ClaudeClient, OpenAIClient, ToolDefinition
from src.core.llm.base import ClaudeModel

async def main():
    # 1. Initialize Client (API key resolves from environment)
    client = ClaudeClient()
    
    # 2. Simple Generation
    print("--- Simple Generation ---")
    response = await client.generate(
        prompt="Explain the core concept of async event loops in Python.",
        model=ClaudeModel.CLAUDE_3_5_SONNET,
        temperature=0.2
    )
    print(f"Response: {response.text}")
    print(f"Cost: ${response.cost_usd:.6f} (Tokens: In={response.input_tokens}, Out={response.output_tokens})")
    print(f"Stop Reason: {response.stop_reason}\n")
    
    # 3. Streaming Generation
    print("--- Streaming Generation ---")
    async for chunk in client.generate_stream(
        prompt="Write a short haiku about speed.",
        model=ClaudeModel.CLAUDE_3_5_HAIKU,
        temperature=0.7
    ):
        print(chunk.text, end="", flush=True)
        if chunk.is_final:
            print(f"\nFinal Cost: ${chunk.cost_usd:.6f}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## System Architecture

```text
       +-------------------------------------------------------------+
       |                         Application                         |
       +------------------------------+------------------------------+
                                      |
                                      | calls async methods
                                      v
       +-------------------------------------------------------------+
       |                       BaseLLMClient                         |
       +-------+-----------------------------+-----------------------+
               |                                     |
               | extends abstract                    | extends abstract
               v                                     v
+-----------------------------+       +------------------------------+
|        ClaudeClient         |       |         OpenAIClient         |
+--------------+--------------+       +--------------+---------------+
               |                                     |
               | calls                               | calls
               v                                     v
     ( Anthropic Async SDK )               ( OpenAI Async Completion )
               |                                     |
               +------------------+------------------+
                                  | maps outputs to
                                  v
+--------------------------------------------------------------------+
|                            LLMResponse                             |
|  - text: str                                                       |
|  - input_tokens / output_tokens: int                               |
|  - cost_usd: float (derived via calculate_cost)                    |
|  - tool_calls: list[ToolCall]                                      |
|  - stop_reason: str (e.g. 'end_turn' / 'max_tokens')                |
+--------------------------------------------------------------------+
```

## What I'd Do Differently (Senior Retrospective)
While this client provides a lightweight, unified interface, scaling it to support agentic production workflows warrants a few key structural refinements:

1. **Declarative Pydantic Tooling / Structured Outputs:** Instead of parsing raw JSON-schema dictionaries for `ToolDefinition`, I would leverage a library like `Instructor` or custom Pydantic-based response mappings to enforce strict, schema-validated outputs. This prevents runtime JSON parsing errors during complex agent tool-calls.
2. **Middleware Pipeline for Hooks (Observability / Caching):** I'd introduce a middleware/interceptor chain (similar to gRPC interceptors). This would decouple cross-cutting concerns like semantic caching (e.g. Redis), request validation, prompt security sanitization, and database auditing without polluting the `generate()` call loop.
3. **Dynamic Pricing Provider:** Currently, model prices are hardcoded inside `tracker.py`. In production, these should be loaded from a config service, a remote registry, or cached from a database/API. This prevents codebase redeployments when vendors adjust prices or introduce dynamic caching discounts (like Claude's prompt caching).
4. **Provider-Agnostic Context Window Management:** Currently, if context limits are exceeded, client calls simply fail at the API level. I would implement an optional, configurable `ContextManager` wrapper that transparently handles message truncating, summary compilation, or token sliding windows for chat sessions.
5. **Parallel Tool Execution:** The multi-turn loop executes tool calls sequentially. In high-concurrency systems, parallel tool calls returned by the model should be resolved concurrently via `asyncio.gather(*tasks)` to minimize roundtrip latencies.
6. **Path & Environment Sandboxing:** The filesystem and python execution tools run on the host system. For security, we should run python code in containerized/WASM environments and sandbox file storage paths.

---

## Tool Registry & Loop Design Decisions

For detailed design decisions about the registry pattern, termination conditions, and graceful tool error handling, refer to [tool_client.md](file:///Users/vamsi_cheruku/Desktop/Agentic%20AI%20Journey/src/core/llm/tool_client.md).

