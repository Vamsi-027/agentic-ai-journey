import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.core.llm import (
    get_llm_client,
    ClaudeClient,
    OpenAIClient,
    ChatSession,
    LLMProvider,
    ClaudeModel,
    OpenAIModel,
    ToolDefinition,
    ToolCall,
    LLMResponse
)

# -----------------------------------------------------------------------------
# 1. Test Claude Client generate()
# -----------------------------------------------------------------------------
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_claude_generate(mock_async_anthropic, mock_anthropic_response):
    """Verify that ClaudeClient.generate calls Messages API and returns correct LLMResponse."""
    mock_client = mock_anthropic_response(text="Hello, human!", input_tokens=10, output_tokens=20)
    mock_async_anthropic.return_value = mock_client
    
    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    response = await client.generate(prompt="Hi")
    
    assert response.text == "Hello, human!"
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.cost_usd > 0.0
    mock_client.messages.create.assert_called_once()


# -----------------------------------------------------------------------------
# 2. Test Claude Client generate_stream()
# -----------------------------------------------------------------------------
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_claude_generate_stream(mock_async_anthropic, mock_anthropic_response):
    """Verify that ClaudeClient.generate_stream streams tokens and sends metadata in the final block."""
    mock_client = mock_anthropic_response(stream_chunks=["A", " B", " C"], input_tokens=15, output_tokens=5)
    mock_async_anthropic.return_value = mock_client
    
    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    
    chunks = []
    async for chunk in client.generate_stream(prompt="Stream please"):
        chunks.append(chunk)
        
    assert len(chunks) == 4  # 3 chunks for tokens + 1 final metadata chunk
    assert chunks[0].text == "A"
    assert chunks[1].text == " B"
    assert chunks[2].text == " C"
    
    # Final chunk verification
    final_chunk = chunks[3]
    assert final_chunk.text == ""
    assert final_chunk.input_tokens == 15
    assert final_chunk.output_tokens == 5
    assert final_chunk.cost_usd > 0.0


# -----------------------------------------------------------------------------
# 3. Test Claude Client generate_with_tools()
# -----------------------------------------------------------------------------
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_claude_generate_with_tools(mock_async_anthropic, mock_tool_use_response):
    """Verify that ClaudeClient.generate_with_tools formats tools and parses tool_use blocks."""
    mock_client = mock_tool_use_response(
        tool_id="tool_weather_123", 
        tool_name="get_weather", 
        tool_input={"city": "Paris"},
        text="Looking up weather...",
        input_tokens=25,
        output_tokens=35
    )
    mock_async_anthropic.return_value = mock_client
    
    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Lookup weather for a city",
        input_schema={"type": "object"}
    )
    
    response = await client.generate_with_tools(
        prompt="What is the weather like in Paris?",
        tools=[weather_tool]
    )
    
    assert response.text == "Looking up weather..."
    assert len(response.tool_calls) == 1
    
    tool_call = response.tool_calls[0]
    assert tool_call.id == "tool_weather_123"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "Paris"}
    
    assert response.input_tokens == 25
    assert response.output_tokens == 35


# -----------------------------------------------------------------------------
# 4. Test OpenAI Client generate_with_tools()
# -----------------------------------------------------------------------------
@patch("src.core.llm.openai.AsyncOpenAI")
async def test_openai_generate_with_tools_success(mock_async_openai):
    """Verify that OpenAIClient.generate_with_tools formats tools and parses tool calls."""
    # 1. Setup mock structures
    mock_tc = MagicMock()
    mock_tc.id = "call_openai_123"
    mock_tc.function.name = "test_tool"
    mock_tc.function.arguments = '{"arg": "value"}'
    
    mock_message = MagicMock()
    mock_message.content = "Opening file..."
    mock_message.tool_calls = [mock_tc]
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"
    
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 12
    mock_usage.completion_tokens = 18
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_async_openai.return_value = mock_client
    
    # 2. Initialize and call
    client = get_llm_client(provider=LLMProvider.OPENAI, api_key="test-api-key")
    tool = ToolDefinition(name="test_tool", description="desc", input_schema={"type": "object"})
    
    response = await client.generate_with_tools(prompt="Test", tools=[tool])
    
    # 3. Assertions
    assert response.text == "Opening file..."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_openai_123"
    assert response.tool_calls[0].name == "test_tool"
    assert response.tool_calls[0].arguments == {"arg": "value"}
    assert response.input_tokens == 12
    assert response.output_tokens == 18


# -----------------------------------------------------------------------------
# 5. Test LLM Client Factory Function
# -----------------------------------------------------------------------------
@patch("src.core.llm.claude.AsyncAnthropic")
@patch("src.core.llm.openai.AsyncOpenAI")
def test_get_llm_client_factory(mock_openai, mock_anthropic):
    """Verify factory initializes correct client types with strings or Enums."""
    mock_openai.return_value = MagicMock()
    mock_anthropic.return_value = MagicMock()
    
    # 1. String arguments
    c1 = get_llm_client(provider="claude")
    assert isinstance(c1, ClaudeClient)
    
    c2 = get_llm_client(provider="anthropic")
    assert isinstance(c2, ClaudeClient)
    
    o1 = get_llm_client(provider="openai")
    assert isinstance(o1, OpenAIClient)
    
    # 2. Enum arguments
    c3 = get_llm_client(provider=LLMProvider.CLAUDE)
    assert isinstance(c3, ClaudeClient)
    
    o2 = get_llm_client(provider=LLMProvider.OPENAI)
    assert isinstance(o2, OpenAIClient)
    
    # 3. Invalid providers raise ValueError
    with pytest.raises(ValueError) as exc_info:
        get_llm_client(provider="unsupported-model")
    assert "Unsupported provider" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 6. Test Chat Session History Management
# -----------------------------------------------------------------------------
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_chat_session_history(mock_async_anthropic, mock_anthropic_response):
    """Verify ChatSession updates conversation history correctly and accepts model Enums."""
    mock_client = mock_anthropic_response(text="Calculus is interesting.", input_tokens=10, output_tokens=15)
    mock_async_anthropic.return_value = mock_client
    
    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    session = ChatSession(
        client=client,
        system_prompt="You are a helper.",
        model=ClaudeModel.CLAUDE_3_5_SONNET
    )
    
    response = await session.send_message("Tell me about calculus.")
    
    assert response.text == "Calculus is interesting."
    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "Tell me about calculus."}
    assert session.history[1] == {"role": "assistant", "content": "Calculus is interesting."}
    
    session.clear_history()
    assert len(session.history) == 0
