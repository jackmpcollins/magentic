from enum import Enum
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(Enum):
    LITELLM = "litellm"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAGENTIC_")

    backend: Backend = Backend.OPENAI
    litellm_model: str = "gpt-3.5-turbo"
    litellm_api_base: str | None = None
    litellm_max_tokens: int | None = None
    litellm_temperature: float | None = None
    openai_model: str = "gpt-3.5-turbo"
    openai_api_key: str | None = None
    openai_api_type: Literal["openai", "azure"] = "openai"
    openai_base_url: str | None = None
    openai_max_tokens: int | None = None
    openai_seed: int | None = None
    openai_temperature: float | None = None


def get_settings() -> Settings:
    return Settings()
