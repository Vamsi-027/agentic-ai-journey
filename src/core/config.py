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

    @property
    def WORKSPACE_ROOT(self) -> str:
        return os.getenv("WORKSPACE_ROOT", os.path.abspath(os.getcwd()))

    @property
    def TAVILY_API_KEY(self) -> str | None:
        return os.getenv("TAVILY_API_KEY")

settings = Settings()
