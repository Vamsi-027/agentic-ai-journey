import backoff
from anthropic import Anthropic, RateLimitError, APIStatusError
from src.core.config import settings


class MessagesWrapper:
    """Wraps Anthropic messages endpoints to provide automatic rate-limit and API status retries."""

    def __init__(self, client: Anthropic):
        self._client = client

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError),
        max_tries=3,
        jitter=backoff.full_jitter,
    )
    def create(self, *args, **kwargs):
        """Creates a message with automated backoff retry logic on rate limits or API errors."""
        return self._client.messages.create(*args, **kwargs)


class AgenticClient:
    """A professional wrapper for the AI client, providing drop-in compatibility with the official SDK."""

    def __init__(self):
        self._anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.messages = MessagesWrapper(self._anthropic_client)
