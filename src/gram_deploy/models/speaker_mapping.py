"""Speaker mapping entity - connects raw speaker IDs to resolved Persons."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ResolutionMethod(str, Enum):
    """How the speaker was identified."""
    VOICE_MATCH = "voice_match"
    CONTEXT_INFERENCE = "context_inference"
    MANUAL = "manual"
    UNRESOLVED = "unresolved"


class SpeakerMapping(BaseModel):
    """Maps a raw speaker ID from transcription to a resolved Person.

    Used to maintain consistent speaker identity across sources.
    """

    raw_speaker_id: str = Field(
        ..., description="Speaker ID as assigned by transcription service"
    )
    deployment_id: str
    source_id: str
    resolved_person_id: Optional[str] = Field(
        None, description="Person ID if resolved, None if unidentified"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    method: ResolutionMethod = Field(default=ResolutionMethod.UNRESOLVED)
    verified: bool = Field(default=False, description="Whether a human verified this mapping")

    # Evidence for the mapping
    evidence_utterances: list[str] = Field(
        default_factory=list,
        description="Utterance IDs that support this mapping"
    )
    evidence_notes: Optional[str] = Field(
        None, description="Explanation of how mapping was determined"
    )

    @property
    def is_resolved(self) -> bool:
        """Whether this speaker has been identified."""
        return self.resolved_person_id is not None

    @property
    def full_raw_id(self) -> str:
        """Full speaker ID including source context."""
        return f"speaker:{self.source_id}/{self.raw_speaker_id}"
