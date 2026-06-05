import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.core.llm import (
    get_llm_client,
    ClaudeClient,
    LLMProvider,
    ToolDefinition,
    ToolCall,
    LLMResponse
)
from conftest import MockMessageResponse, MockToolUseBlock, MockTextBlock

# ==============================================================================
# Helper to construct multiple content block messages
# ==============================================================================
def make_mock_message_response(content_blocks, input_tokens=10, output_tokens=20, stop_reason="tool_use"):
    return MockMessageResponse(
        content=content_blocks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason
    )


# ==============================================================================
# 1. Test standard successful tool loop execution
# ==============================================================================
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_tool_called_correctly(mock_async_anthropic):
    """Verify standard tool loop runs: invokes tool, feeds result back, terminates."""
    # First call: LLM returns tool_use block
    tool_use_block = MockToolUseBlock(
        id="call_abc_123",
        name="test_tool",
        input_dict={"param": "world"}
    )
    first_resp = make_mock_message_response([tool_use_block], input_tokens=10, output_tokens=15, stop_reason="tool_use")
    
    # Second call: LLM returns final text answer
    text_block = MockTextBlock("Result matches: Hello world")
    second_resp = make_mock_message_response([text_block], input_tokens=8, output_tokens=12, stop_reason="end_turn")
    
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[first_resp, second_resp])
    mock_async_anthropic.return_value = mock_client

    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    
    # Register mock tool
    tool_def = ToolDefinition(
        name="test_tool",
        description="A simple test tool",
        input_schema={"type": "object"}
    )
    def my_test_tool(param: str) -> str:
        return f"Hello {param}"
        
    client.register_tool(tool_def, my_test_tool)
    
    # Run loop
    response = await client.run_with_tools(prompt="Run tool please")
    
    # Asserts
    assert response.text == "Result matches: Hello world"
    assert response.input_tokens == 18  # 10 + 8
    assert response.output_tokens == 27  # 15 + 12
    assert response.cost_usd > 0.0
    assert response.stop_reason == "end_turn"
    
    # Verify Messages create was called exactly twice
    assert mock_client.messages.create.call_count == 2
    
    # Verify second messages create call received the correct history containing tool result
    call_args_list = mock_client.messages.create.call_args_list
    second_call_messages = call_args_list[1][1]["messages"]
    
    # History should contain initial prompt, assistant tool call response, and user tool result response
    assert len(second_call_messages) == 3
    assert second_call_messages[0] == {"role": "user", "content": "Run tool please"}
    
    # Assistant message should have tool call block
    assistant_content = second_call_messages[1]["content"]
    assert len(assistant_content) == 1
    assert assistant_content[0]["type"] == "tool_use"
    assert assistant_content[0]["id"] == "call_abc_123"
    
    # User message should have tool result block
    user_content = second_call_messages[2]["content"]
    assert len(user_content) == 1
    assert user_content[0]["type"] == "tool_result"
    assert user_content[0]["tool_use_id"] == "call_abc_123"
    assert user_content[0]["content"] == "Hello world"
    assert user_content[0]["is_error"] is False


# ==============================================================================
# 2. Test tool execution error handled gracefully
# ==============================================================================
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_tool_error_handled_gracefully(mock_async_anthropic):
    """Verify tool execution exceptions are caught and passed back as is_error=True blocks."""
    # First call: LLM returns tool_use block
    tool_use_block = MockToolUseBlock(
        id="call_err_456",
        name="fail_tool",
        input_dict={}
    )
    first_resp = make_mock_message_response([tool_use_block], input_tokens=12, output_tokens=14, stop_reason="tool_use")
    
    # Second call: LLM returns text noting error
    text_block = MockTextBlock("Noted the error.")
    second_resp = make_mock_message_response([text_block], input_tokens=9, output_tokens=11, stop_reason="end_turn")
    
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[first_resp, second_resp])
    mock_async_anthropic.return_value = mock_client

    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    
    # Register error-throwing tool
    tool_def = ToolDefinition(name="fail_tool", description="Fails", input_schema={})
    def my_failing_tool() -> str:
        raise ValueError("Something went wrong internally")
        
    client.register_tool(tool_def, my_failing_tool)
    
    # Run loop
    response = await client.run_with_tools(prompt="Do it")
    
    assert response.text == "Noted the error."
    assert mock_client.messages.create.call_count == 2
    
    # Verify error payload was passed back gracefully to the LLM
    call_args_list = mock_client.messages.create.call_args_list
    second_call_messages = call_args_list[1][1]["messages"]
    
    user_content = second_call_messages[2]["content"]
    assert user_content[0]["type"] == "tool_result"
    assert user_content[0]["tool_use_id"] == "call_err_456"
    assert "ValueError" in user_content[0]["content"] or "Something went wrong" in user_content[0]["content"]
    assert user_content[0]["is_error"] is True


# ==============================================================================
# 3. Test parallel tool execution
# ==============================================================================
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_parallel_tools_executed(mock_async_anthropic):
    """Verify multiple tool calls in a single turn are both dispatched and sent back."""
    # First call: LLM returns 2 tool_use blocks
    tool_use_1 = MockToolUseBlock(id="id_1", name="tool_a", input_dict={"val": 5})
    tool_use_2 = MockToolUseBlock(id="id_2", name="tool_b", input_dict={"val": 10})
    first_resp = make_mock_message_response([tool_use_1, tool_use_2], input_tokens=15, output_tokens=22, stop_reason="tool_use")
    
    # Second call: LLM returns final text answer
    text_block = MockTextBlock("Both tools executed.")
    second_resp = make_mock_message_response([text_block], input_tokens=10, output_tokens=15, stop_reason="end_turn")
    
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[first_resp, second_resp])
    mock_async_anthropic.return_value = mock_client

    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    
    # Register tools
    tool_def_a = ToolDefinition(name="tool_a", description="A", input_schema={})
    tool_def_b = ToolDefinition(name="tool_b", description="B", input_schema={})
    
    client.register_tool(tool_def_a, lambda val: f"A:{val}")
    client.register_tool(tool_def_b, lambda val: f"B:{val}")
    
    # Run loop
    response = await client.run_with_tools(prompt="Run both")
    
    assert response.text == "Both tools executed."
    assert mock_client.messages.create.call_count == 2
    
    # Verify second messages create call has both tool result content blocks
    call_args_list = mock_client.messages.create.call_args_list
    second_call_messages = call_args_list[1][1]["messages"]
    
    user_content = second_call_messages[2]["content"]
    assert len(user_content) == 2
    assert user_content[0]["tool_use_id"] == "id_1"
    assert user_content[0]["content"] == "A:5"
    assert user_content[1]["tool_use_id"] == "id_2"
    assert user_content[1]["content"] == "B:10"


# ==============================================================================
# 4. Test unknown tool raises KeyError
# ==============================================================================
@patch("src.core.llm.claude.AsyncAnthropic")
async def test_unknown_tool_raises_exception(mock_async_anthropic):
    """Verify that an unknown tool name returned by the LLM raises a KeyError."""
    # First call: LLM returns tool_use block with unregistered name
    tool_use_block = MockToolUseBlock(
        id="call_xyz_789",
        name="unknown_tool",
        input_dict={}
    )
    first_resp = make_mock_message_response([tool_use_block], input_tokens=10, output_tokens=15, stop_reason="tool_use")
    
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=first_resp)
    mock_async_anthropic.return_value = mock_client

    client = get_llm_client(provider=LLMProvider.CLAUDE, api_key="test-api-key")
    
    # We do NOT register "unknown_tool" but do register another tool so tool-loop runs
    tool_def = ToolDefinition(name="other_tool", description="Other", input_schema={})
    client.register_tool(tool_def, lambda: "other")
    
    # Run loop and check for KeyError exception
    with pytest.raises(KeyError) as exc_info:
        await client.run_with_tools(prompt="Trigger unknown")
        
    assert "unknown_tool" in str(exc_info.value)
