"""Person entity - a known team member who may appear in deployments."""

from datetime import datetime
from typing import Optional
import re

from pydantic import Field, model_validator

from gram_deploy.models.base import GRAMModel, utc_now


class VoiceSample(GRAMModel):
    """A reference audio sample for voice matching."""

    source_id: str
    start_time: float = Field(..., ge=0, description="Start time in source-local seconds")
    end_time: float = Field(..., ge=0, description="End time in source-local seconds")
    verified: bool = Field(default=False, description="Whether a human verified this sample")

    @model_validator(mode="after")
    def validate_time_range(self) -> "VoiceSample":
        """Ensure start_time <= end_time."""
        if self.start_time > self.end_time:
            raise ValueError("start_time must be <= end_time")
        return self


class Person(GRAMModel):
    """A known team member for speaker identification.

    ID format: person:{slug}
    Example: person:damion
    """

    id: str = Field(..., pattern=r"^person:[a-z0-9_]+$")
    name: str = Field(..., description="Full display name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names or nicknames")
    role: Optional[str] = Field(None, description="Job title or role")
    voice_samples: list[VoiceSample] = Field(default_factory=list)
    voice_embedding: Optional[list[float]] = Field(
        None, description="Computed voice embedding for speaker identification"
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def generate_id(cls, name: str) -> str:
        """Generate person ID from name."""
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        # Use first name only if slug is too long
        if len(slug) > 20:
            slug = slug.split("_")[0]
        return f"person:{slug}"

    @classmethod
    def from_name(cls, name: str, role: Optional[str] = None) -> "Person":
        """Create a Person from a name."""
        return cls(
            id=cls.generate_id(name),
            name=name,
            role=role,
        )

    def matches_name(self, name: str) -> bool:
        """Check if a name matches this person (including aliases)."""
        name_lower = name.lower()
        if name_lower == self.name.lower():
            return True
        if name_lower in [a.lower() for a in self.aliases]:
            return True
        # Check first name match
        first_name = self.name.split()[0].lower()
        if name_lower == first_name:
            return True
        return False
