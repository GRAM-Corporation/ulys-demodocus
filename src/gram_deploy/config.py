"""Runtime configuration for GRAM Deployment Processing System."""

import os
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class TranscriptionProvider(str, Enum):
    """Supported transcription providers."""
    ELEVENLABS = "elevenlabs"
    ASSEMBLYAI = "assemblyai"
    DEEPGRAM = "deepgram"


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Data directory
    data_dir: Path = Field(
        default=Path("./deployments"),
        validation_alias="GRAM_DATA_DIR"
    )

    # Transcription
    transcription_provider: TranscriptionProvider = Field(
        default=TranscriptionProvider.ELEVENLABS,
        validation_alias="GRAM_TRANSCRIPTION_PROVIDER"
    )
    elevenlabs_api_key: str | None = Field(
        default=None,
        validation_alias="ELEVENLABS_API_KEY"
    )
    assemblyai_api_key: str | None = Field(
        default=None,
        validation_alias="ASSEMBLYAI_API_KEY"
    )
    deepgram_api_key: str | None = Field(
        default=None,
        validation_alias="DEEPGRAM_API_KEY"
    )

    # Semantic analysis
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY"
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        validation_alias="GRAM_ANTHROPIC_MODEL"
    )

    # Processing settings
    max_concurrent_transcriptions: int = Field(
        default=3,
        validation_alias="GRAM_MAX_CONCURRENT_TRANSCRIPTIONS"
    )
    alignment_confidence_threshold: float = Field(
        default=0.8,
        validation_alias="GRAM_ALIGNMENT_THRESHOLD"
    )
    speaker_confidence_threshold: float = Field(
        default=0.7,
        validation_alias="GRAM_SPEAKER_THRESHOLD"
    )

    # S3 settings for presigned URL transcription
    s3_bucket: str | None = Field(
        default=None,
        validation_alias="GRAM_S3_BUCKET"
    )
    s3_region: str = Field(
        default="us-east-1",
        validation_alias="AWS_REGION"
    )

    # Search settings
    search_db_path: Path | None = Field(
        default=None,
        validation_alias="GRAM_SEARCH_DB"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    def get_transcription_api_key(self) -> str | None:
        """Get the API key for the configured transcription provider."""
        match self.transcription_provider:
            case TranscriptionProvider.ELEVENLABS:
                return self.elevenlabs_api_key
            case TranscriptionProvider.ASSEMBLYAI:
                return self.assemblyai_api_key
            case TranscriptionProvider.DEEPGRAM:
                return self.deepgram_api_key

    def get_deployment_path(self, deployment_id: str) -> Path:
        """Get the filesystem path for a deployment."""
        # Convert deploy:20250119_vinci_01 -> deploy_20250119_vinci_01
        dir_name = deployment_id.replace(":", "_")
        return self.data_dir / dir_name

    def get_search_db(self) -> Path:
        """Get the path to the search database."""
        if self.search_db_path:
            return self.search_db_path
        return self.data_dir / "search.db"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
