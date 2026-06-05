from typing import List, Dict, Optional, AsyncGenerator, Generator
from src.core.llm.base import BaseLLMClient, LLMResponse

class ChatSession:
    """Manages conversation history and generates responses using an underlying BaseLLMClient."""

    def __init__(
        self,
        client: BaseLLMClient,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []

    async def send_message(self, message: str) -> LLMResponse:
        """Sends a user message asynchronously, updates chat history, and returns the response."""
        self.history.append({"role": "user", "content": message})
        
        response = await self.client.generate(
            prompt=self.history,
            system_prompt=self.system_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        self.history.append({"role": "assistant", "content": response.text})
        return response

    async def send_message_stream(self, message: str) -> AsyncGenerator[LLMResponse, None]:
        """Sends a user message asynchronously, streaming tokens and logging the final assistant response."""
        self.history.append({"role": "user", "content": message})
        
        text_accumulated = ""
        async for chunk in self.client.generate_stream(
            prompt=self.history,
            system_prompt=self.system_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ):
            if chunk.text:
                text_accumulated += chunk.text
            yield chunk
            
        self.history.append({"role": "assistant", "content": text_accumulated})


    def clear_history(self):
        """Clears the session's conversation history."""
        self.history.clear()
