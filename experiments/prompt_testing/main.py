from src.core.client import AgenticClient

# Initialize the centralized core client, which handles secrets, logging, and rate-limit retries
client = AgenticClient()

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Explain transformers simply."}],
)

print(message.content[0].text)
