"""Data models for the GRAM Deployment Processing System.

All entities use Pydantic for validation and serialization.
ID formats:
- Deployment: deploy:{YYYYMMDD}_{location}_{sequence}
- Source: source:{deployment_id}/{device_type}_{device_number}
- Person: person:{slug}
- Transcript: transcript:source:{...}
- Utterance: utterance:{deployment_id}/{uuid}
- Event: event:{deployment_id}/{uuid}
- ActionItem: action:{deployment_id}/{uuid}
- Insight: insight:{deployment_id}/{uuid}
"""

from gram_deploy.models.base import GRAMModel, utc_now, ensure_utc
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.source import Source, SourceFile, DeviceType, TranscriptStatus
from gram_deploy.models.transcript import (
    RawTranscript,
    TranscriptSegment,
    TranscriptSpeaker,
    WordTiming,
)
from gram_deploy.models.person import Person, VoiceSample
from gram_deploy.models.speaker_mapping import SpeakerMapping, ResolutionMethod
from gram_deploy.models.canonical_utterance import (
    CanonicalUtterance,
    CanonicalWord,
    UtteranceSource,
    get_utterances_in_range,
)
from gram_deploy.models.event import DeploymentEvent, EventType, Severity, ExtractionMethod
from gram_deploy.models.action_item import ActionItem, ActionItemStatus, Priority
from gram_deploy.models.insight import DeploymentInsight, InsightType, TimeRange, SupportingEvidence
from gram_deploy.models.timeline import TimelineSegment, TimeAlignment, SourceAlignment, CrossCorrelation

__all__ = [
    # Base
    "GRAMModel",
    "utc_now",
    "ensure_utc",
    # Deployment
    "Deployment",
    "DeploymentStatus",
    # Source
    "Source",
    "SourceFile",
    "DeviceType",
    "TranscriptStatus",
    # Transcript
    "RawTranscript",
    "TranscriptSegment",
    "TranscriptSpeaker",
    "WordTiming",
    # Person
    "Person",
    "VoiceSample",
    # Speaker Mapping
    "SpeakerMapping",
    "ResolutionMethod",
    # Canonical Utterance
    "CanonicalUtterance",
    "CanonicalWord",
    "UtteranceSource",
    "get_utterances_in_range",
    # Event
    "DeploymentEvent",
    "EventType",
    "Severity",
    "ExtractionMethod",
    # Action Item
    "ActionItem",
    "ActionItemStatus",
    "Priority",
    # Insight
    "DeploymentInsight",
    "InsightType",
    "TimeRange",
    "SupportingEvidence",
    # Timeline
    "TimelineSegment",
    "TimeAlignment",
    "SourceAlignment",
    "CrossCorrelation",
]
