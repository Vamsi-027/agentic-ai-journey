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
    assert '"arg1": "string"' in prompt
    assert '"arg2": "integer"' in prompt


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