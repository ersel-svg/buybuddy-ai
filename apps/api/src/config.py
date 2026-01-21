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
    runpod_endpoint_preview: str = ""  # Segmentation preview worker
    runpod_endpoint_od_annotation: str = ""  # OD AI annotation worker
    runpod_endpoint_od_training: str = ""  # OD model training worker

    # External APIs
    gemini_api_key: str = ""
    hf_token: str = ""

    # Buybuddy Legacy API
    buybuddy_api_url: str = "https://api-legacy.buybuddy.co/api/v1"
    buybuddy_username: str = ""
    buybuddy_password: str = ""

    # Qdrant Vector Database
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Slack Integration
    slack_webhook_url: str = ""

    # Roboflow Integration (optional - users can provide their own API key)
    roboflow_api_key: str = ""

    # Roboflow Import Checkpoint Settings
    roboflow_import_dir: str = "./data/roboflow_imports"  # Persistent storage for ZIP files
    roboflow_import_max_age_hours: int = 48  # Cleanup completed/failed imports after this time

    # Processing defaults
    target_resolution: int = 518

    # CORS - allow all origins for now (can be restricted later)
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
