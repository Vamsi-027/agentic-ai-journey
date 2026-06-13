import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.agent.react import ReActAgent, parse_react_action, AgentResult, REACT_SYSTEM_PROMPT_TEMPLATE
from src.core.llm import BaseLLMClient, ToolDefinition, LLMResponse

# ==============================================================================
# 1. Test parse_react_action
# ==============================================================================
def test_parse_react_action_single_line():
    text = 'Thought: Let\'s write a file.\nAction: write_file {"path": "test.txt", "content": "hello"}'
    status, tool, args = parse_react_action(text)
    assert status == "action"
    assert tool == "write_file"
    assert args == {"path": "test.txt", "content": "hello"}

def test_parse_react_action_multi_line():
    text = (
        "Thought: Let's write a file with multi-line JSON.\n"
        "Action: write_file {\n"
        '  "path": "test.txt",\n'
        '  "content": "hello"\n'
        "}"
    )
    status, tool, args = parse_react_action(text)
    assert status == "action"
    assert tool == "write_file"
    assert args == {"path": "test.txt", "content": "hello"}

def test_parse_react_action_final_answer():
    text = "Thought: I am done.\nFinal Answer: Task is complete!"
    status, val, args = parse_react_action(text)
    assert status == "final"
    assert val == "Task is complete!"
    assert args == {}

def test_parse_react_action_invalid_json():
    text = 'Thought: Let\'s try.\nAction: write_file {"path": "test.txt", "content": hello}'
    status, val, args = parse_react_action(text)
    assert status == "error"
    assert "Failed to parse Action arguments" in val
    assert args["raw_args"] == '{"path": "test.txt", "content": hello}'

def test_parse_react_action_none():
    text = "Thought: What should I do?\nMaybe I should just wait."
    status, val, args = parse_react_action(text)
    assert status == "none"


# ==============================================================================
# 2. Test system prompt dynamic building
# ==============================================================================
def test_dynamic_system_prompt_builder():
    mock_client = MagicMock(spec=BaseLLMClient)
    tool1 = ToolDefinition(name="tool_a", description="Description A", input_schema={"properties": {"arg1": {"type": "string"}}})
    tool2 = ToolDefinition(name="tool_b", description="Description B", input_schema={"properties": {"arg2": {"type": "integer"}}})
    
    mock_client.registry = {
        "tool_a": (tool1, lambda: None),
        "tool_b": (tool2, lambda: None)
    }
    
    agent = ReActAgent(client=mock_client)
    prompt = agent._build_system_prompt()
    
    assert "tool_a: Description A" in prompt
    assert "tool_b: Description B" in prompt
    assert "arg1 (optional): string" in prompt
    assert "arg2 (optional): integer" in prompt


# ==============================================================================
# 3. Test ReActAgent execution loop and AgentResult
# ==============================================================================
@pytest.mark.asyncio
async def test_react_agent_run_success():
    # Mock LLM Client
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    
    # Define tool
    tool_def = ToolDefinition(name="calc_tool", description="Calculates", input_schema={"properties": {"expr": {"type": "string"}}})
    mock_client.registry["calc_tool"] = (tool_def, lambda expr: f"result of {expr}")
    
    # Mock LLM Client dispatch
    mock_client.dispatch = AsyncMock(return_value="result of 2+2")
    
    # Mock LLM responses:
    # 1. Action: call calc_tool
    resp1 = LLMResponse(
        text='Thought: Let\'s calculate 2+2.\nAction: calc_tool {"expr": "2+2"}',
        model="test-model",
        input_tokens=10,
        output_tokens=15,
        cost_usd=0.01,
        stop_reason="end_turn"
    )
    # 2. Final Answer:
    resp2 = LLMResponse(
        text="Thought: I have calculated it.\nFinal Answer: The result is 4.",
        model="test-model",
        input_tokens=20,
        output_tokens=25,
        cost_usd=0.02,
        stop_reason="end_turn"
    )
    mock_client.chat = AsyncMock(side_effect=[resp1, resp2])
    
    agent = ReActAgent(client=mock_client, model="test-model")
    result = await agent.run("Calculate 2+2")
    
    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.total_steps == 2
    assert result.answer == "The result is 4."
    
    # Verify steps
    assert len(result.steps) == 2
    assert result.steps[0]["step"] == 1
    assert "Let's calculate 2+2." in result.steps[0]["thought"]
    assert result.steps[0]["action"] == 'calc_tool {"expr": "2+2"}'
    assert result.steps[0]["observation"] == "result of 2+2"
    
    assert result.steps[1]["step"] == 2
    assert "I have calculated it." in result.steps[1]["thought"]
    assert result.steps[1]["action"] == "Final Answer: The result is 4."
    assert result.steps[1]["observation"] == ""


@pytest.mark.asyncio
async def test_react_agent_unregistered_tool():
    mock_client = MagicMock(spec=BaseLLMClient)
    # Register only tool_a
    tool_def = ToolDefinition(name="tool_a", description="A", input_schema={})
    mock_client.registry = {
        "tool_a": (tool_def, lambda: None)
    }
    
    # Client dispatch raises KeyError for tool_b
    mock_client.dispatch = AsyncMock(side_effect=KeyError("tool_b not found"))
    
    # Mock LLM responses:
    # 1. Action: call unregistered tool_b
    resp1 = LLMResponse(
        text='Thought: Let\'s call tool_b.\nAction: tool_b {}',
        model="test-model",
        input_tokens=10,
        output_tokens=15,
        cost_usd=0.01,
        stop_reason="end_turn"
    )
    # 2. Final Answer
    resp2 = LLMResponse(
        text="Thought: I cannot call tool_b.\nFinal Answer: Tool tool_b is unregistered.",
        model="test-model",
        input_tokens=20,
        output_tokens=25,
        cost_usd=0.02,
        stop_reason="end_turn"
    )
    
    mock_client.chat = AsyncMock(side_effect=[resp1, resp2])
    
    agent = ReActAgent(client=mock_client)
    result = await agent.run("Call tool_b")
    
    assert result.success is True
    assert result.total_steps == 2
    # The first step should have an observation with the available tools list
    assert "Available tools: ['tool_a']" in result.steps[0]["observation"]


# ==============================================================================
# 4. Test ReActAgent Tracing & SQLite Log Integration
# ==============================================================================

@pytest.mark.asyncio
async def test_react_agent_tracing_success(tmp_path):
    import sqlite3
    db_path = tmp_path / "test_trace.db"
    
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    tool_def = ToolDefinition(name="calc_tool", description="Calculates", input_schema={})
    mock_client.registry["calc_tool"] = (tool_def, lambda: "res")
    mock_client.dispatch = AsyncMock(return_value="result of 2+2")
    
    resp1 = LLMResponse(
        text='Thought: Let\'s calculate.\nAction: calc_tool {"expr": "2+2"}',
        model="test-model",
        input_tokens=10,
        output_tokens=15,
        cost_usd=0.01,
        stop_reason="end_turn"
    )
    resp2 = LLMResponse(
        text="Thought: I have calculated.\nFinal Answer: 4",
        model="test-model",
        input_tokens=20,
        output_tokens=25,
        cost_usd=0.02,
        stop_reason="end_turn"
    )
    mock_client.chat = AsyncMock(side_effect=[resp1, resp2])
    
    with patch("src.core.database.DEFAULT_DB_PATH", str(db_path)):
        agent = ReActAgent(client=mock_client, model="test-model")
        result = await agent.run("Calculate 2+2")
        
        assert result.success is True
        
        # Connect to test db and verify records
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check agent_runs
        cursor.execute("SELECT run_id, task, model, outcome, total_tokens, total_cost_usd, started_at, finished_at FROM agent_runs")
        runs = cursor.fetchall()
        assert len(runs) == 1
        run_id, task, model, outcome, total_tokens, total_cost_usd, started_at, finished_at = runs[0]
        assert task == "Calculate 2+2"
        assert model == "test-model"
        assert outcome == "success"
        assert total_tokens == 70  # (10+15) + (20+25)
        assert total_cost_usd == pytest.approx(0.03)
        assert started_at is not None
        assert finished_at is not None
        
        # Check agent_steps
        cursor.execute("SELECT step_num, thought, action, action_input, observation, tokens_used FROM agent_steps ORDER BY step_num")
        steps = cursor.fetchall()
        assert len(steps) == 2
        
        # Step 1
        assert steps[0][0] == 1
        assert "Let's calculate." in steps[0][1]
        assert steps[0][2] == "calc_tool"
        assert "2+2" in steps[0][3]
        assert steps[0][4] == "result of 2+2"
        assert steps[0][5] == 25
        
        # Step 2
        assert steps[1][0] == 2
        assert "I have calculated." in steps[1][1]
        assert steps[1][2] == "Final Answer"
        assert "4" in steps[1][3]
        assert steps[1][5] == 45
        
        conn.close()


@pytest.mark.asyncio
async def test_react_agent_tracing_exception_safety():
    import sqlite3
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    mock_client.chat = AsyncMock(return_value=LLMResponse(
        text="Final Answer: Done",
        model="test-model",
        input_tokens=5,
        output_tokens=5,
        cost_usd=0.0,
        stop_reason="end_turn"
    ))
    
    # Force sqlite3.connect to raise an error to simulate broken database
    with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("Mocked DB error")):
        agent = ReActAgent(client=mock_client, model="test-model")
        # Running the agent should complete successfully without raising DB exception
        result = await agent.run("Do something")
        assert result.success is True
        assert result.answer == "Done"


# ==============================================================================
# 5. Test ReActAgent Reflexion Loop & limits
# ==============================================================================

@pytest.mark.asyncio
async def test_react_agent_reflexion_success():
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    
    loop1_resp = LLMResponse(text="Final Answer: Attempt 1 answer (wrong)", model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    eval1_resp = LLMResponse(text='{"success": false, "reason": "Incorrect math"}', model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    critique_resp = LLMResponse(text="I need to verify subtraction.", model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    loop2_resp = LLMResponse(text="Final Answer: Attempt 2 answer (correct)", model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    eval2_resp = LLMResponse(text='{"success": true, "reason": "Correct math"}', model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    
    mock_client.chat = AsyncMock(side_effect=[
        loop1_resp,
        eval1_resp,
        critique_resp,
        loop2_resp,
        eval2_resp
    ])
    
    agent = ReActAgent(client=mock_client, model="test-model")
    with patch("src.core.database.DEFAULT_DB_PATH", ":memory:"):
        result = await agent.run_with_reflection("Calculate subtraction", max_attempts=3)
        
        assert result.success is True
        assert result.answer == "Attempt 2 answer (correct)"
        assert mock_client.chat.call_count == 5


@pytest.mark.asyncio
async def test_react_agent_reflexion_cap():
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    
    loop_resp = LLMResponse(text="Final Answer: Wrong answer", model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    eval_resp = LLMResponse(text='{"success": false, "reason": "Incorrect"}', model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    critique_resp = LLMResponse(text="Self critique content", model="test", input_tokens=5, output_tokens=5, cost_usd=0.0)
    
    mock_client.chat = AsyncMock(side_effect=[
        loop_resp,       # Attempt 1 run
        eval_resp,       # Attempt 1 eval
        critique_resp,   # Attempt 1 critique
        loop_resp,       # Attempt 2 run
        eval_resp,       # Attempt 2 eval
        critique_resp,   # Attempt 2 critique
        loop_resp,       # Attempt 3 run
        eval_resp        # Attempt 3 eval
    ])
    
    agent = ReActAgent(client=mock_client, model="test-model")
    with patch("src.core.database.DEFAULT_DB_PATH", ":memory:"):
        result = await agent.run_with_reflection("Calculate math", max_attempts=3)
        
        assert result.success is False
        assert result.answer == "Wrong answer"
        assert mock_client.chat.call_count == 8


# ==============================================================================
# 6. Test New Enhancements: robust JSON parsing, loop detection, safety evaluation
# ==============================================================================

def test_parse_react_action_nested_dict():
    # Verify that action parsing successfully extracts nested JSON objects/dict literals
    text = (
        "Thought: Let's edit the file.\n"
        "Action: edit_file "
        r"""{
  "path": "src/core/agent/react.py",
  "old_str": "def test():\n  return {'a': 1}",
  "new_str": "def test():\n  return {'a': 2}"
}"""
    )
    status, tool, args = parse_react_action(text)
    assert status == "action"
    assert tool == "edit_file"
    assert args == {
        "path": "src/core/agent/react.py",
        "old_str": "def test():\n  return {'a': 1}",
        "new_str": "def test():\n  return {'a': 2}"
    }


@pytest.mark.asyncio
async def test_react_agent_loop_detection():
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    
    # Register tool
    tool_def = ToolDefinition(name="read_file", description="reads file", input_schema={"properties": {"path": {"type": "string"}}})
    mock_client.registry["read_file"] = (tool_def, lambda path: "content")
    
    # Mock tool execution return
    mock_client.dispatch = AsyncMock(return_value="file content")
    
    # LLM responses: Action read_file (step 1), Action read_file again (step 2), Final Answer
    resp1 = LLMResponse(text='Thought: Read file.\nAction: read_file {"path": "test.txt"}', model="test", input_tokens=0, output_tokens=0, cost_usd=0.0)
    resp2 = LLMResponse(text='Thought: Read file again.\nAction: read_file {"path": "test.txt"}', model="test", input_tokens=0, output_tokens=0, cost_usd=0.0)
    resp3 = LLMResponse(text="Thought: I have the answer.\nFinal Answer: Done", model="test", input_tokens=0, output_tokens=0, cost_usd=0.0)
    mock_client.chat = AsyncMock(side_effect=[resp1, resp2, resp3])
    
    agent = ReActAgent(client=mock_client, model="test-model", max_steps=5)
    with patch("src.core.database.DEFAULT_DB_PATH", ":memory:"):
        result = await agent.run("Check file contents")
        
        assert result.success is True
        assert len(result.steps) == 3
        # Verify the second step observation contains the loop detection error
        assert "Error: You already called 'read_file' with these exact arguments" in result.steps[1]["observation"]
        # Verify the first step actually executed
        assert result.steps[0]["observation"] == "file content"


@pytest.mark.asyncio
async def test_generate_reflection_empty_steps():
    from src.core.agent.react import generate_reflection
    mock_client = MagicMock(spec=BaseLLMClient)
    critique = await generate_reflection(mock_client, "Calculate math", steps=[])
    assert "failed before completing any steps" in critique
    assert "list_directory" in critique


@pytest.mark.asyncio
async def test_evaluate_success_llm_judge_fallback():
    from src.core.agent.react import evaluate_success
    mock_client = MagicMock(spec=BaseLLMClient)
    # Mock LLM judge throwing an exception (e.g. timeout)
    mock_client.chat = AsyncMock(side_effect=Exception("Timeout"))
    
    success, reason = await evaluate_success(mock_client, "Math task", "4", model="test")
    assert success is False
    assert "assumed failure to be safe" in reason


@pytest.mark.asyncio
async def test_evaluate_success_with_surrounding_prose():
    from src.core.agent.react import evaluate_success
    mock_client = MagicMock(spec=BaseLLMClient)
    
    # LLM judge outputs text before/after JSON
    judge_response = LLMResponse(
        text=(
            "Here is my evaluation:\n"
            "```json\n"
            '{"success": true, "reason": "Math is fully correct."}\n'
            "```\n"
            "Hope this helps!"
        ),
        model="test",
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0
    )
    mock_client.chat = AsyncMock(return_value=judge_response)
    
    success, reason = await evaluate_success(mock_client, "Math task", "4", model="test")
    assert success is True
    assert reason == "Math is fully correct."


@pytest.mark.asyncio
async def test_react_agent_with_rag_pipeline():
    from src.core.retrieval.rag_pipeline import RAGPipeline
    from src.core.llm import RETRIEVE_CONTEXT_TOOL
    
    # Mock LLM Client
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.registry = {}
    
    # Store registration
    def mock_register_tool(tool_def, func):
        mock_client.registry[tool_def.name] = (tool_def, func)
    mock_client.register_tool = mock_register_tool
    
    # Mock chat response
    resp = LLMResponse(
        text="Thought: Let's stop.\nFinal Answer: Complete.",
        model="test-model",
        input_tokens=10,
        output_tokens=15,
        cost_usd=0.01,
        stop_reason="end_turn"
    )
    mock_client.chat = AsyncMock(return_value=resp)
    
    # Mock RAGPipeline
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.index_directory = AsyncMock(return_value=(42, 0.00084))
    mock_pipeline.vector_store = MagicMock()
    
    agent = ReActAgent(
        client=mock_client,
        model="test-model",
        rag_pipeline=mock_pipeline
    )
    
    # Assert retrieve_context is registered
    assert RETRIEVE_CONTEXT_TOOL.name in mock_client.registry
    
    # Run agent loop
    result = await agent.run("Verify retrieval works.")
    
    # Assert index_directory was called
    mock_pipeline.index_directory.assert_called_once()
    assert result.success is True