from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(Enum):
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAGENTIC_")

    backend: Backend = Backend.OPENAI
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int | None = None
    openai_temperature: float | None = None


def get_settings() -> Settings:
    return Settings()
