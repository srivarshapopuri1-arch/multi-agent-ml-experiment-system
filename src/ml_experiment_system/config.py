from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    random_state: int = 42
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
