import backoff
import openai
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI

from src.core.config import settings
from src.core.llm.base import BaseLLMClient, LLMResponse, OpenAIModel, ToolDefinition, ToolCall
from src.core.llm.tracker import calculate_cost

class OpenAIClient(BaseLLMClient):
    """Async LLM Client wrapping the OpenAI API. Includes backoff retry and token cost tracking."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        from src.core.llm.base import LLMProvider
        # Allow passing custom key, fallback to loaded settings
        self.client = AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.default_model = OpenAIModel.GPT_4O.value
        self.provider = LLMProvider.OPENAI

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter
    )
    async def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Asynchronously call OpenAI Chat Completion API, returning a unified response."""
        target_model = model or self.default_model
        if hasattr(target_model, "value"):
            target_model = target_model.value
        
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs = {
            "model": target_model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if stop:
            kwargs["stop"] = stop

        response = await self.client.chat.completions.create(**kwargs)
        
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calculate_cost(target_model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            model=target_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            is_final=True,
            stop_reason=response.choices[0].finish_reason
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Asynchronously call OpenAI Chat Completion API, returning a unified response."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter
    )
    async def _start_stream(self, kwargs):
        """Helper to invoke client completions stream with backoff decorator."""
        return await self.client.chat.completions.create(**kwargs)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously streams chunks from the OpenAI API, yielding final usage statistics at the end."""
        target_model = model or self.default_model
        if hasattr(target_model, "value"):
            target_model = target_model.value
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}  # Requests usage metrics in final chunk
        }

        stream = await self._start_stream(kwargs)

        finish_reason = None
        async for chunk in stream:
            # Stage 1: Yield text tokens if choices are present in the chunk
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield LLMResponse(
                        text=delta.content,
                        model=target_model,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0
                    )
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            
            # Stage 2: Extract usage metadata and yield the final pricing block
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                cost = calculate_cost(target_model, input_tokens, output_tokens)
                
                yield LLMResponse(
                    text="",
                    model=target_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    is_final=True,
                    stop_reason=finish_reason
                )

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter
    )
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Asynchronously call OpenAI completions API with tools."""
        target_model = model or self.default_model
        if hasattr(target_model, "value"):
            target_model = target_model.value
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": prompt})

        formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for tool in tools
        ]

        response = await self.client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=formatted_tools
        )
        
        message = response.choices[0].message
        text = message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calculate_cost(target_model, input_tokens, output_tokens)

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                import json
                try:
                    arguments = json.loads(tc.function.arguments)
                except Exception:
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments
                    )
                )

        return LLMResponse(
            text=text,
            model=target_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            tool_calls=tool_calls,
            is_final=True,
            stop_reason=response.choices[0].finish_reason
        )
