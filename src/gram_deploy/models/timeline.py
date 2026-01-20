"""Timeline entities - time alignment and visualization support."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from gram_deploy.models.base import GRAMModel, utc_now


class TimelineSegment(GRAMModel):
    """A contiguous period of recording from a source.

    Used for timeline visualization to show coverage.
    """

    source_id: str
    canonical_start_ms: int = Field(..., ge=0)
    canonical_end_ms: int = Field(..., ge=0)
    file_index: int = Field(..., ge=0, description="Index into the source's files array")
    gap_before_ms: Optional[int] = Field(
        None, ge=0, description="Duration of gap before this segment, None if first"
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "TimelineSegment":
        """Ensure canonical_start_ms <= canonical_end_ms."""
        if self.canonical_start_ms > self.canonical_end_ms:
            raise ValueError("canonical_start_ms must be <= canonical_end_ms")
        return self

    @property
    def duration_ms(self) -> int:
        """Duration of this segment in milliseconds."""
        return self.canonical_end_ms - self.canonical_start_ms


@dataclass
class SourceAlignment:
    """Alignment data for a single source."""

    source_id: str
    offset_ms: int  # Add to source-local time to get canonical time
    confidence: float  # 0.0 to 1.0
    method: str  # audio_fingerprint, visual_sync, metadata, manual, unaligned


@dataclass
class CrossCorrelation:
    """Cross-correlation result between two sources."""

    source_a_id: str
    source_b_id: str
    offset_ms: int  # source_b - source_a offset
    confidence: float
    method: str


class TimeAlignment(GRAMModel):
    """Complete time alignment for a deployment.

    Maps all sources to a canonical timeline.
    """

    deployment_id: str
    canonical_start_time: datetime = Field(
        ..., description="Absolute wall-clock time of canonical zero"
    )
    canonical_end_time: Optional[datetime] = Field(
        None, description="End of canonical timeline"
    )
    source_offsets: dict[str, int] = Field(
        default_factory=dict, description="source_id -> offset in ms"
    )
    confidence_scores: dict[str, float] = Field(
        default_factory=dict, description="source_id -> alignment confidence"
    )
    alignment_methods: dict[str, str] = Field(
        default_factory=dict, description="source_id -> method used"
    )
    cross_correlations: list[dict] = Field(
        default_factory=list,
        description="Pairs with their offsets: [{source_a, source_b, offset_ms, confidence}]"
    )

    # Metadata
    computed_at: datetime = Field(default_factory=utc_now)
    issues: list[str] = Field(
        default_factory=list, description="Alignment issues or warnings"
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "TimeAlignment":
        """Ensure canonical_start_time <= canonical_end_time if both are set."""
        if self.canonical_start_time and self.canonical_end_time:
            if self.canonical_start_time > self.canonical_end_time:
                raise ValueError("canonical_start_time must be <= canonical_end_time")
        return self

    def get_offset(self, source_id: str) -> int:
        """Get the canonical offset for a source."""
        return self.source_offsets.get(source_id, 0)

    def get_confidence(self, source_id: str) -> float:
        """Get the alignment confidence for a source."""
        return self.confidence_scores.get(source_id, 0.0)

    def source_to_canonical(self, source_id: str, local_ms: int) -> int:
        """Convert source-local time to canonical time."""
        return local_ms + self.get_offset(source_id)

    def canonical_to_source(self, source_id: str, canonical_ms: int) -> int:
        """Convert canonical time to source-local time."""
        return canonical_ms - self.get_offset(source_id)
