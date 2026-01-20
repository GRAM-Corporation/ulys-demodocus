"""Deployment entity - top-level container for a field deployment session."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator
import re


class DeploymentStatus(str, Enum):
    """Processing status of a deployment."""
    INGESTING = "ingesting"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"


class Deployment(BaseModel):
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Processing checkpoints for resumption
    checkpoint: Optional[str] = Field(None, description="Last completed processing step")
    error_message: Optional[str] = Field(None, description="Error message if status is FAILED")

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

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

    def model_post_init(self, __context) -> None:
        """Update timestamp on modification."""
        self.updated_at = datetime.utcnow()
