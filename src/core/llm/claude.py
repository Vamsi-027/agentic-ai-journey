import backoff
from typing import AsyncGenerator, Optional
from anthropic import AsyncAnthropic, RateLimitError, APIStatusError

from src.core.config import settings
from src.core.llm.base import BaseLLMClient, LLMResponse, ClaudeModel, ToolDefinition, ToolCall
from src.core.llm.tracker import calculate_cost


class ClaudeClient(BaseLLMClient):
    """Async LLM Client wrapping the Anthropic API. Includes backoff retry and token cost tracking."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        from src.core.llm.base import LLMProvider
        self.client = AsyncAnthropic(api_key=api_key or settings.ANTHROPIC_API_KEY)
        self.default_model = ClaudeModel.CLAUDE_3_5_SONNET.value
        self.provider = LLMProvider.CLAUDE

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter,
    )
    async def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Asynchronously call Claude Chat Completion API with conversation history."""
        target_model = model or self.default_model
        if hasattr(target_model, "value"):
            target_model = target_model.value

        kwargs = {
            "model": target_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(target_model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            model=target_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            is_final=True,
            stop_reason=response.stop_reason,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Asynchronously call Claude API, resolving parameters and returning a unified response."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter,
    )
    async def _start_stream(self, kwargs):
        """Helper to invoke client stream context with backoff decorator."""
        return self.client.messages.stream(**kwargs)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously streams chunks from the Claude API, yielding final usage statistics at the end."""
        target_model = model or self.default_model
        if hasattr(target_model, "value"):
            target_model = target_model.value

        kwargs = {
            "model": target_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        stream_mgr = await self._start_stream(kwargs)

        async with stream_mgr as stream:
            async for text_chunk in stream.text_stream:
                yield LLMResponse(
                    text=text_chunk,
                    model=target_model,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                )

            # Retrieve final message token counts after stream termination
            final_msg = await stream.get_final_message()
            input_tokens = final_msg.usage.input_tokens
            output_tokens = final_msg.usage.output_tokens
            cost = calculate_cost(target_model, input_tokens, output_tokens)

            # Yield final empty summary chunk carrying cost tracking metadata
            yield LLMResponse(
                text="",
                model=target_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                is_final=True,
                stop_reason=final_msg.stop_reason,
            )

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter,
    )
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Asynchronously call Claude API with tool definitions, parsing tool_use content blocks."""
        target_model = model or self.default_model
        if hasattr(target_model, "value"):
            target_model = target_model.value

        formatted_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in tools
        ]

        kwargs = {
            "model": target_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}],
            "tools": formatted_tools,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)

        text_blocks = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                    )
                )

        text = "\n".join(text_blocks)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(target_model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            model=target_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            tool_calls=tool_calls,
            is_final=True,
            stop_reason=response.stop_reason,
        )
