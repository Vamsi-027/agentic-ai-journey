import asyncio
import os
from src.core.llm import (
    get_llm_client,
    ChatSession,
    LLMProvider,
    ClaudeModel,
    OpenAIModel,
    ToolDefinition,
)


async def run_claude_async_tests(client):
    print("\n--- 1. Testing Claude ASYNC Generate ---")
    response = await client.generate(
        prompt="Explain quantum computing in one short sentence.",
        system_prompt="Be extremely brief.",
        model=ClaudeModel.CLAUDE_HAIKU_4_5,
        temperature=0.0,
    )
    print(f"Model: {response.model}")
    print(f"Response: {response.text.strip()}")
    print(f"Tokens: Input={response.input_tokens}, Output={response.output_tokens}")
    print(f"Cost: ${response.cost_usd:.6f}")

    print("\n--- 2. Testing Claude ASYNC Stream ---")
    print("Stream: ", end="", flush=True)
    async for chunk in client.generate_stream(
        prompt="Count from 1 to 5 with space.",
        system_prompt="Be concise. Just output numbers.",
        model=ClaudeModel.CLAUDE_HAIKU_4_5,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
        if chunk.is_final:
            print(
                f"\nFinal Usage -> In: {chunk.input_tokens}, Out: {chunk.output_tokens}, Cost: ${chunk.cost_usd:.6f}"
            )


async def run_claude_sync_tests(client):
    print("\n--- 3. Testing Claude SYNC Generate ---")
    response = await client.generate(
        prompt="Explain what a compiler does in one sentence.",
        system_prompt="Keep it simple.",
        model=ClaudeModel.CLAUDE_HAIKU_4_5,
    )
    print(f"Response: {response.text.strip()}")
    print(f"Cost: ${response.cost_usd:.6f}")

    print("\n--- 4. Testing Claude SYNC Stream ---")
    # Note: This stream runs without a system prompt guidance (unguided). 
    # Therefore, the model will output a full explanation of primary colors,
    # often including context on paint vs light (CMYK vs RGB) pigments.
    print("Stream: ", end="", flush=True)
    async for chunk in client.generate_stream(
        prompt="List primary colors.", model=ClaudeModel.CLAUDE_HAIKU_4_5
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
        if chunk.is_final:
            print(f"\nFinal Usage -> Cost: ${chunk.cost_usd:.6f}")


async def run_chat_session_tests(client):
    print("\n--- 5. Testing Multi-Turn Chat Session ---")
    session = ChatSession(
        client=client,
        system_prompt="You are a friendly mathematician.",
        model=ClaudeModel.CLAUDE_HAIKU_4_5,
    )

    print("User: What is 5 + 5?")
    res1 = await session.send_message("What is 5 + 5?")
    print(f"Assistant: {res1.text.strip()} (Cost: ${res1.cost_usd:.6f})")

    print("User: Multiply that answer by 2.")
    res2 = await session.send_message("Multiply that answer by 2.")
    print(f"Assistant: {res2.text.strip()} (Cost: ${res2.cost_usd:.6f})")

    print("\nChat History Logged:")
    for msg in session.history:
        print(f"  [{msg['role'].upper()}]: {msg['content']}")


async def run_openai_tests():
    print("\n--- 6. Testing OpenAI Integration ---")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or "your-openai" in openai_key:
        print("⚠️ OPENAI_API_KEY not set. Skipping OpenAI client execution.")
        return

    try:
        openai_client = get_llm_client(provider=LLMProvider.OPENAI)

        # Async Generate
        response = await openai_client.generate(
            prompt="Translate 'hello world' to French.",
            model=OpenAIModel.GPT_4O_MINI,
            temperature=0.0,
        )
        print(f"OpenAI Async Response: {response.text.strip()}")
        print(f"Cost: ${response.cost_usd:.6f}")

        # Async Stream
        print("OpenAI Stream: ", end="", flush=True)
        async for chunk in openai_client.generate_stream(
            prompt="Translate 'goodbye' to Spanish.", model=OpenAIModel.GPT_4O_MINI
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            if chunk.input_tokens > 0:
                print(f"\nOpenAI Stream Cost: ${chunk.cost_usd:.6f}")

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")


async def run_tool_calling_tests(claude_client, openai_client):
    print("\n--- 7. Testing Tool Calling on Claude ---")
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. Seattle, WA",
                }
            },
            "required": ["city"],
        },
    )

    try:
        response = await claude_client.generate_with_tools(
            prompt="What is the weather like in Paris?",
            tools=[weather_tool],
            model=ClaudeModel.CLAUDE_HAIKU_4_5,
        )
        print(f"Text Response: {response.text.strip()}")
        print(f"Tool Calls Count: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"  Tool Call ID: {tc.id}")
            print(f"  Tool Call Name: {tc.name}")
            print(f"  Tool Call Arguments: {tc.arguments}")
    except Exception as e:
        print(f"❌ Claude tool calling test failed: {e}")

    print("\n--- 8. Testing Tool Calling Stub on OpenAI ---")
    try:
        await openai_client.generate_with_tools(
            prompt="What is the weather like in Seattle?",
            tools=[weather_tool],
            model=OpenAIModel.GPT_4O_MINI,
        )
        print(
            "❌ Expected NotImplementedError from OpenAI tool calling, but got no error."
        )
    except NotImplementedError as e:
        print(f"✅ OpenAI NotImplementedError successfully raised: {e}")
    except Exception as e:
        print(f"❌ OpenAI tool calling raised unexpected error: {e}")


async def main():
    print("🚀 Starting Unified LLM Client Wrapper Verification Tests...")

    # Get Claude client
    try:
        claude_client = get_llm_client(provider=LLMProvider.CLAUDE)

        # Run Claude tests
        await run_claude_async_tests(claude_client)
        await run_claude_sync_tests(claude_client)
        await run_chat_session_tests(claude_client)

    except Exception as e:
        print(f"❌ Claude tests failed: {e}")

    # Run OpenAI tests
    await run_openai_tests()

    # Run Tool Calling tests
    try:
        claude_client = get_llm_client(provider=LLMProvider.CLAUDE)
        openai_client = get_llm_client(provider=LLMProvider.OPENAI)
        await run_tool_calling_tests(claude_client, openai_client)
    except Exception as e:
        print(f"❌ Tool calling setup failed: {e}")

    print("\n🎉 Verification Completed.")


if __name__ == "__main__":
    # asyncio.run() is the correct entry point for sync callers.
    # Never call async methods directly from sync code — use this pattern only
    # at the outermost boundary (CLI scripts, __main__ blocks).
    asyncio.run(main())
