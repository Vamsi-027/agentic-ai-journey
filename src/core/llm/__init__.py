from typing import Optional, Union
from src.core.llm.base import (
    BaseLLMClient,
    LLMResponse,
    LLMProvider,
    ClaudeModel,
    OpenAIModel,
    ToolDefinition,
    ToolCall
)
from src.core.llm.claude import ClaudeClient
from src.core.llm.openai import OpenAIClient
from src.core.llm.chat import ChatSession

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "LLMProvider",
    "ClaudeModel",
    "OpenAIModel",
    "ToolDefinition",
    "ToolCall",
    "ClaudeClient",
    "OpenAIClient",
    "ChatSession",
    "get_llm_client"
]

def get_llm_client(
    provider: Union[str, LLMProvider],
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """Factory function to retrieve a configured LLM client instance by provider name or enum."""
    provider_name = provider.value if isinstance(provider, LLMProvider) else str(provider)
    provider_clean = provider_name.lower().strip()
    
    if provider_clean in (LLMProvider.CLAUDE.value, LLMProvider.ANTHROPIC.value):
        return ClaudeClient(api_key=api_key, **kwargs)
    elif provider_clean == LLMProvider.OPENAI.value:
        return OpenAIClient(api_key=api_key, **kwargs)
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Please choose a valid LLMProvider or string (e.g., 'claude' or 'openai')."
        )

