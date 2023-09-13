from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAGENTIC_")

    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float | None = None


def get_settings() -> Settings:
    return Settings()
