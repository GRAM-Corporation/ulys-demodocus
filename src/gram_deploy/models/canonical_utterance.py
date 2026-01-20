"""Canonical utterance entity - speech segment in the unified transcript."""

from typing import Optional
import uuid

from pydantic import BaseModel, Field


class UtteranceSource(BaseModel):
    """Reference to the source of an utterance."""

    source_id: str
    local_start_time: float = Field(..., description="Start time in source-local seconds")
    local_end_time: float = Field(..., description="End time in source-local seconds")
    raw_segment_index: Optional[int] = Field(
        None, description="Index in the RawTranscript segments array"
    )


class CanonicalWord(BaseModel):
    """Word-level timing in canonical time."""

    text: str
    canonical_start_ms: int
    canonical_end_ms: int


class CanonicalUtterance(BaseModel):
    """A speech segment in the unified canonical transcript.

    Positioned on the canonical timeline with resolved speaker identity.

    ID format: utterance:{deployment_id}/{uuid}
    """

    id: str = Field(..., pattern=r"^utterance:.+$")
    deployment_id: str
    text: str
    canonical_start_ms: int = Field(..., description="Start in ms from canonical zero")
    canonical_end_ms: int = Field(..., description="End in ms from canonical zero")
    speaker_id: Optional[str] = Field(None, description="Resolved person ID, None if unidentified")
    speaker_confidence: float = Field(0.0, ge=0.0, le=1.0)
    sources: list[UtteranceSource] = Field(
        default_factory=list, description="Which sources captured this utterance"
    )
    words: Optional[list[CanonicalWord]] = None
    is_duplicate: bool = Field(
        default=False,
        description="True if captured by multiple sources and merged"
    )

    @classmethod
    def generate_id(cls, deployment_id: str) -> str:
        """Generate utterance ID."""
        return f"utterance:{deployment_id}/{uuid.uuid4().hex[:12]}"

    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return self.canonical_end_ms - self.canonical_start_ms

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0

    def overlaps_with(self, other: "CanonicalUtterance", min_overlap_ratio: float = 0.5) -> bool:
        """Check if this utterance overlaps significantly with another."""
        overlap_start = max(self.canonical_start_ms, other.canonical_start_ms)
        overlap_end = min(self.canonical_end_ms, other.canonical_end_ms)
        overlap_duration = max(0, overlap_end - overlap_start)

        min_duration = min(self.duration_ms, other.duration_ms)
        if min_duration == 0:
            return False

        return (overlap_duration / min_duration) >= min_overlap_ratio
