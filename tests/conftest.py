import pytest
from unittest.mock import AsyncMock, MagicMock

class MockUsage:
    def __init__(self, input_tokens=10, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class MockTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text

class MockToolUseBlock:
    def __init__(self, id, name, input_dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input_dict

class MockMessageResponse:
    def __init__(self, content, input_tokens=15, output_tokens=25, stop_reason="end_turn"):
        self.content = content
        self.usage = MockUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason

class MockStreamManager:
    def __init__(self, text_chunks, input_tokens=12, output_tokens=18):
        self.text_chunks = text_chunks
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def text_stream(self):
        async def generator():
            for chunk in self.text_chunks:
                yield chunk
        return generator()

    async def get_final_message(self):
        return MockMessageResponse([], self.input_tokens, self.output_tokens, stop_reason="end_turn")


@pytest.fixture
def mock_anthropic_response():
    """Fixture to mock a standard text response and text streaming from Anthropic."""
    def _creator(text="Mocked Claude Response", input_tokens=15, output_tokens=25, stream_chunks=None, stop_reason="end_turn"):
        mock_client = MagicMock()
        
        # Mock generate() endpoint: messages.create
        content = [MockTextBlock(text)]
        message_resp = MockMessageResponse(content, input_tokens, output_tokens, stop_reason)
        mock_client.messages.create = AsyncMock(return_value=message_resp)
        
        # Mock generate_stream() endpoint: messages.stream
        chunks = stream_chunks or ["Mocked", " stream", " response"]
        stream_mgr = MockStreamManager(chunks, input_tokens, output_tokens)
        mock_client.messages.stream = MagicMock(return_value=stream_mgr)
        
        return mock_client
    return _creator


@pytest.fixture
def mock_tool_use_response():
    """Fixture to mock a tool call response from Anthropic."""
    def _creator(tool_id="tool_123", tool_name="get_weather", tool_input=None, text="", input_tokens=20, output_tokens=30, stop_reason="tool_use"):
        mock_client = MagicMock()
        
        input_data = tool_input or {"city": "Paris"}
        content = []
        if text:
            content.append(MockTextBlock(text))
        content.append(MockToolUseBlock(tool_id, tool_name, input_data))
        
        message_resp = MockMessageResponse(content, input_tokens, output_tokens, stop_reason)
        mock_client.messages.create = AsyncMock(return_value=message_resp)
        
        return mock_client
    return _creator
