import os
from dotenv import load_dotenv

# Load environment variables by walking up the directory tree to locate the root .env file
load_dotenv()

class Settings:
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please check your root .env file."
            )
        return key

    @property
    def OPENAI_API_KEY(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please check your root .env file."
            )
        return key

settings = Settings()
