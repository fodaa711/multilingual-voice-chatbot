from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    WHISPER_MODEL: str = "base"

    LLM_PROVIDER: Literal["groq", "openai", "ollama", "anthropic"] = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    TTS_ENGINE: Literal["gtts", "bark", "edge-tts"] = "gtts"

    SYSTEM_PROMPT: str = (
        "You are a helpful multilingual assistant. "
        "Always reply in the SAME language the user spoke. "
        "Keep answers short — 2 to 3 sentences."
    )


settings = Settings()