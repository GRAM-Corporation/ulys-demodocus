"""Deployment insight entity - semantic observations extracted from content."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class InsightType(str, Enum):
    """Type of deployment insight."""
    TECHNICAL_OBSERVATION = "technical_observation"
    PROCESS_IMPROVEMENT = "process_improvement"
    RISK_IDENTIFIED = "risk_identified"
    SUCCESS_FACTOR = "success_factor"
    RESOURCE_CONSTRAINT = "resource_constraint"
    DECISION_RATIONALE = "decision_rationale"
    LESSON_LEARNED = "lesson_learned"
    CUSTOM = "custom"


class SupportingEvidence(BaseModel):
    """Evidence supporting an insight."""

    utterance_id: str
    quote: str


class TimeRange(BaseModel):
    """A time range in canonical milliseconds."""

    start_ms: int
    end_ms: int


class DeploymentInsight(BaseModel):
    """A semantic observation extracted from deployment content.

    ID format: insight:{deployment_id}/{uuid}
    """

    id: str = Field(..., pattern=r"^insight:.+$")
    deployment_id: str
    insight_type: InsightType
    content: str = Field(..., description="The insight text")
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    time_range: Optional[TimeRange] = Field(
        None, description="Time range this insight applies to"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Additional metadata
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    related_events: list[str] = Field(
        default_factory=list, description="Event IDs related to this insight"
    )

    @classmethod
    def generate_id(cls, deployment_id: str) -> str:
        """Generate insight ID."""
        return f"insight:{deployment_id}/{uuid.uuid4().hex[:12]}"
