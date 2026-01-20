"""Source entity - a single video recording device and its captured footage."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Type of recording device."""
    GOPRO = "gopro"
    PHONE = "phone"
    FIXED = "fixed"
    DRONE = "drone"
    OTHER = "other"


class TranscriptStatus(str, Enum):
    """Transcription processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class SourceFile(BaseModel):
    """A single video file within a source."""

    filename: str
    file_path: str
    file_size_bytes: Optional[int] = None
    duration_seconds: float
    start_offset_ms: int = Field(
        ..., description="Milliseconds from canonical start when this file begins"
    )
    end_offset_ms: Optional[int] = Field(
        None, description="Milliseconds from canonical start when this file ends"
    )
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None

    def model_post_init(self, __context) -> None:
        """Compute end_offset if not set."""
        if self.end_offset_ms is None:
            self.end_offset_ms = self.start_offset_ms + int(self.duration_seconds * 1000)


class Source(BaseModel):
    """A video recording source (camera) and its files.

    ID format: source:{deployment_id}/{device_type}_{device_number}
    Example: source:deploy:20250119_vinci_01/gopro_01
    """

    id: str = Field(..., pattern=r"^source:deploy:\d{8}_[a-z0-9_]+_\d{2}/[a-z]+_\d{2}$")
    deployment_id: str
    device_type: DeviceType
    device_number: int = Field(..., ge=1)
    device_model: Optional[str] = Field(None, description="Specific model, e.g., 'GoPro Hero 12'")
    operator: Optional[str] = Field(None, description="Person ID of camera operator")
    files: list[SourceFile] = Field(default_factory=list)
    total_duration_seconds: Optional[float] = None
    canonical_offset_ms: int = Field(
        0, description="Milliseconds to add to source-local time to get canonical time"
    )
    alignment_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence in time alignment"
    )
    alignment_method: str = Field(
        "unaligned",
        description="How alignment was determined: audio_fingerprint, visual_sync, metadata, manual, unaligned"
    )
    transcript_status: TranscriptStatus = Field(default=TranscriptStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def generate_id(cls, deployment_id: str, device_type: DeviceType, device_number: int) -> str:
        """Generate a source ID from components."""
        return f"source:{deployment_id}/{device_type.value}_{device_number:02d}"

    @property
    def total_duration(self) -> float:
        """Compute total duration from files if not explicitly set."""
        if self.total_duration_seconds is not None:
            return self.total_duration_seconds
        return sum(f.duration_seconds for f in self.files)

    def local_to_canonical(self, local_ms: int) -> int:
        """Convert source-local timestamp to canonical timestamp."""
        return local_ms + self.canonical_offset_ms

    def canonical_to_local(self, canonical_ms: int) -> int:
        """Convert canonical timestamp to source-local timestamp."""
        return canonical_ms - self.canonical_offset_ms
