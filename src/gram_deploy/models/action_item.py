"""Action item entity - tasks and follow-ups extracted from deployment dialogue."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from pydantic import Field

from gram_deploy.models.base import GRAMModel, utc_now


class ActionItemStatus(str, Enum):
    """Status of an action item."""
    EXTRACTED = "extracted"  # Just extracted, not reviewed
    CONFIRMED = "confirmed"  # Human confirmed it's a real action item
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISMISSED = "dismissed"  # Not actually an action item


class Priority(str, Enum):
    """Priority level for action items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionItem(GRAMModel):
    """A task or follow-up extracted from deployment dialogue.

    ID format: action:{deployment_id}/{uuid}
    """

    id: str = Field(..., pattern=r"^action:.+$")
    deployment_id: str
    description: str = Field(..., description="What needs to be done (imperative form)")
    source_utterance_id: str = Field(..., description="Utterance where this was mentioned")
    canonical_time_ms: int = Field(..., ge=0, description="When this was mentioned")
    mentioned_by: Optional[str] = Field(None, description="Person ID who mentioned it")
    assigned_to: Optional[str] = Field(None, description="Person ID if assignee was stated")
    deadline: Optional[datetime] = Field(None, description="Deadline if one was stated")
    priority: Optional[Priority] = None
    status: ActionItemStatus = Field(default=ActionItemStatus.EXTRACTED)
    extraction_confidence: float = Field(0.0, ge=0.0, le=1.0)
    verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Context
    context_quote: Optional[str] = Field(
        None, description="The exact quote that triggered extraction"
    )
    category: Optional[str] = Field(
        None, description="Category like 'equipment', 'process', 'follow-up'"
    )
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def generate_id(cls, deployment_id: str) -> str:
        """Generate action item ID."""
        return f"action:{deployment_id}/{uuid.uuid4().hex[:12]}"
