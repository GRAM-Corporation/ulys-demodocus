"""Deployment event entity - significant moments on the canonical timeline."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from pydantic import Field

from gram_deploy.models.base import GRAMModel, utc_now


class EventType(str, Enum):
    """Type of deployment event."""
    DEPLOYMENT_START = "deployment_start"
    DEPLOYMENT_END = "deployment_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    MILESTONE = "milestone"
    ISSUE = "issue"
    DECISION = "decision"
    OBSERVATION = "observation"
    ACTION_ITEM = "action_item"
    CUSTOM = "custom"


class Severity(str, Enum):
    """Severity level for events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ExtractionMethod(str, Enum):
    """How the event was extracted."""
    MANUAL = "manual"
    LLM_EXTRACTED = "llm_extracted"
    RULE_BASED = "rule_based"
    IMPORTED = "imported"


class DeploymentEvent(GRAMModel):
    """A significant event on the deployment timeline.

    ID format: event:{deployment_id}/{uuid}
    """

    id: str = Field(..., pattern=r"^event:.+$")
    deployment_id: str
    event_type: EventType
    canonical_time_ms: int = Field(..., ge=0, description="When the event occurred")
    duration_ms: Optional[int] = Field(
        None, ge=0, description="Duration if event spans time, None for point events"
    )
    title: str = Field(..., max_length=200)
    description: Optional[str] = None
    severity: Optional[Severity] = None
    related_utterances: list[str] = Field(
        default_factory=list, description="IDs of related utterances"
    )
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.LLM_EXTRACTED)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Additional metadata
    phase: Optional[str] = Field(None, description="Deployment phase this event belongs to")
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def generate_id(cls, deployment_id: str) -> str:
        """Generate event ID."""
        return f"event:{deployment_id}/{uuid.uuid4().hex[:12]}"

    @property
    def canonical_end_ms(self) -> int:
        """End time of event (start + duration, or just start for point events)."""
        if self.duration_ms:
            return self.canonical_time_ms + self.duration_ms
        return self.canonical_time_ms
