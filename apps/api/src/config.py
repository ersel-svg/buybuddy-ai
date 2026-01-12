"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_env_file() -> str:
    """Find .env file in current or parent directories."""
    for path in [Path(".env"), Path("../.env"), Path("../../.env")]:
        if path.exists():
            return str(path)
    return ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "BuyBuddy AI API"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # Runpod
    runpod_api_key: str = ""
    runpod_endpoint_video: str = ""
    runpod_endpoint_augmentation: str = ""
    runpod_endpoint_training: str = ""
    runpod_endpoint_embedding: str = ""

    # External APIs
    gemini_api_key: str = ""
    hf_token: str = ""

    # Buybuddy Legacy API
    buybuddy_api_url: str = "https://api.buybuddy.co"
    buybuddy_api_key: str = ""

    # Processing defaults
    target_resolution: int = 518

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
