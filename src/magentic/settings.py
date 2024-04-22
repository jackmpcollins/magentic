from enum import Enum
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(Enum):
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"
    MISTRAL = "mistral"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAGENTIC_")

    backend: Backend = Backend.OPENAI

    anthropic_model: str = "claude-3-opus-20240229"
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_max_tokens: int = 1024
    anthropic_temperature: float | None = None

    litellm_model: str = "gpt-4-turbo"
    litellm_api_base: str | None = None
    litellm_max_tokens: int | None = None
    litellm_temperature: float | None = None

    mistral_model: str = "mistral-large-latest"
    mistral_api_key: str | None = None
    mistral_base_url: str | None = None
    mistral_max_tokens: int | None = None
    mistral_seed: int | None = None
    mistral_temperature: float | None = None

    openai_model: str = "gpt-4-turbo"
    openai_api_key: str | None = None
    openai_api_type: Literal["openai", "azure"] = "openai"
    openai_base_url: str | None = None
    openai_max_tokens: int | None = None
    openai_seed: int | None = None
    openai_temperature: float | None = None


def get_settings() -> Settings:
    return Settings()
