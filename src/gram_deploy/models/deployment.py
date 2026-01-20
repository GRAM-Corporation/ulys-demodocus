"""Deployment entity - top-level container for a field deployment session."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import re

from pydantic import Field, field_validator, model_validator

from gram_deploy.models.base import GRAMModel, utc_now


class DeploymentStatus(str, Enum):
    """Processing status of a deployment."""
    INGESTING = "ingesting"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"


class Deployment(GRAMModel):
    """A field deployment session containing multiple video sources.

    ID format: deploy:{YYYYMMDD}_{location}_{sequence}
    Example: deploy:20250119_vinci_01
    """

    id: str = Field(..., pattern=r"^deploy:\d{8}_[a-z0-9_]+_\d{2}$")
    location: str = Field(..., description="Human-readable location name")
    date: str = Field(..., description="ISO 8601 date (YYYY-MM-DD)")
    canonical_start_time: Optional[datetime] = Field(
        None, description="Absolute wall-clock time of canonical timeline zero"
    )
    canonical_end_time: Optional[datetime] = Field(
        None, description="Absolute wall-clock time of last recorded moment"
    )
    sources: list[str] = Field(default_factory=list, description="Source IDs in this deployment")
    team_members: list[str] = Field(default_factory=list, description="Person IDs of team members present")
    notes: Optional[str] = Field(None, description="Free-form notes")
    status: DeploymentStatus = Field(default=DeploymentStatus.INGESTING)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Processing checkpoints for resumption
    checkpoint: Optional[str] = Field(None, description="Last completed processing step")
    error_message: Optional[str] = Field(None, description="Error message if status is FAILED")

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @model_validator(mode="after")
    def validate_time_range(self) -> "Deployment":
        """Ensure canonical_start_time <= canonical_end_time if both are set."""
        if self.canonical_start_time and self.canonical_end_time:
            if self.canonical_start_time > self.canonical_end_time:
                raise ValueError("canonical_start_time must be <= canonical_end_time")
        return self

    @classmethod
    def generate_id(cls, location: str, date: str, sequence: int = 1) -> str:
        """Generate a deployment ID from components."""
        date_compact = date.replace("-", "")
        location_slug = re.sub(r"[^a-z0-9]+", "_", location.lower()).strip("_")
        return f"deploy:{date_compact}_{location_slug}_{sequence:02d}"

    @property
    def duration_seconds(self) -> Optional[float]:
        """Total deployment duration in seconds."""
        if self.canonical_start_time and self.canonical_end_time:
            return (self.canonical_end_time - self.canonical_start_time).total_seconds()
        return None

    def get_source_path(self, source_id: str, base_path: str | Path = "deployments") -> Path:
        """Get the filesystem path for a source within this deployment.

        Args:
            source_id: The full source ID (e.g., "source:deploy:20250119_vinci_01/gopro_01")
            base_path: Base deployments directory (default: "deployments")

        Returns:
            Path to the source directory

        Raises:
            ValueError: If source_id doesn't belong to this deployment
        """
        # Extract deployment ID from source ID
        # source:{deployment_id}/{device_type}_{device_number}
        if not source_id.startswith("source:"):
            raise ValueError(f"Invalid source ID format: {source_id}")

        parts = source_id[7:].rsplit("/", 1)  # Remove "source:" prefix
        if len(parts) != 2:
            raise ValueError(f"Invalid source ID format: {source_id}")

        source_deploy_id, device_part = parts

        if source_deploy_id != self.id:
            raise ValueError(
                f"Source {source_id} does not belong to deployment {self.id}"
            )

        # Convert deployment ID to directory name
        # deploy:20250119_vinci_01 -> deploy_20250119_vinci_01
        deploy_dir = self.id.replace(":", "_")

        return Path(base_path) / deploy_dir / "sources" / device_part

    def model_post_init(self, __context) -> None:
        """Update timestamp on modification."""
        self.updated_at = utc_now()
