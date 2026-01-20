"""Canonical utterance entity - speech segment in the unified transcript."""

from typing import Optional
import uuid

from pydantic import Field, model_validator

from gram_deploy.models.base import GRAMModel


class UtteranceSource(GRAMModel):
    """Reference to the source of an utterance."""

    source_id: str
    local_start_time: float = Field(..., ge=0, description="Start time in source-local seconds")
    local_end_time: float = Field(..., ge=0, description="End time in source-local seconds")
    raw_segment_index: Optional[int] = Field(
        None, ge=0, description="Index in the RawTranscript segments array"
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "UtteranceSource":
        """Ensure local_start_time <= local_end_time."""
        if self.local_start_time > self.local_end_time:
            raise ValueError("local_start_time must be <= local_end_time")
        return self


class CanonicalWord(GRAMModel):
    """Word-level timing in canonical time."""

    text: str
    canonical_start_ms: int = Field(..., ge=0)
    canonical_end_ms: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "CanonicalWord":
        """Ensure canonical_start_ms <= canonical_end_ms."""
        if self.canonical_start_ms > self.canonical_end_ms:
            raise ValueError("canonical_start_ms must be <= canonical_end_ms")
        return self


class CanonicalUtterance(GRAMModel):
    """A speech segment in the unified canonical transcript.

    Positioned on the canonical timeline with resolved speaker identity.

    ID format: utterance:{deployment_id}/{uuid}
    """

    id: str = Field(..., pattern=r"^utterance:.+$")
    deployment_id: str
    text: str
    canonical_start_ms: int = Field(..., ge=0, description="Start in ms from canonical zero")
    canonical_end_ms: int = Field(..., ge=0, description="End in ms from canonical zero")
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

    @model_validator(mode="after")
    def validate_time_range(self) -> "CanonicalUtterance":
        """Ensure canonical_start_ms <= canonical_end_ms."""
        if self.canonical_start_ms > self.canonical_end_ms:
            raise ValueError("canonical_start_ms must be <= canonical_end_ms")
        return self

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

    def is_in_range(self, start_ms: int, end_ms: int) -> bool:
        """Check if this utterance overlaps with a time range.

        Args:
            start_ms: Start of range in canonical milliseconds
            end_ms: End of range in canonical milliseconds

        Returns:
            True if any part of the utterance falls within the range
        """
        return self.canonical_start_ms < end_ms and self.canonical_end_ms > start_ms


def get_utterances_in_range(
    utterances: list[CanonicalUtterance],
    start_ms: int,
    end_ms: int,
    fully_contained: bool = False
) -> list[CanonicalUtterance]:
    """Get utterances that fall within a time range.

    Args:
        utterances: List of utterances to filter
        start_ms: Start of range in canonical milliseconds
        end_ms: End of range in canonical milliseconds
        fully_contained: If True, only return utterances fully within range.
                        If False, return any utterance that overlaps the range.

    Returns:
        List of utterances in the specified range, sorted by start time
    """
    if fully_contained:
        result = [
            u for u in utterances
            if u.canonical_start_ms >= start_ms and u.canonical_end_ms <= end_ms
        ]
    else:
        result = [u for u in utterances if u.is_in_range(start_ms, end_ms)]

    return sorted(result, key=lambda u: u.canonical_start_ms)
