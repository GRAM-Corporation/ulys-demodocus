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

from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.source import Source, SourceFile, DeviceType, TranscriptStatus
from gram_deploy.models.transcript import RawTranscript, TranscriptSegment, WordTiming
from gram_deploy.models.person import Person, VoiceSample
from gram_deploy.models.speaker_mapping import SpeakerMapping, ResolutionMethod
from gram_deploy.models.canonical_utterance import CanonicalUtterance, UtteranceSource
from gram_deploy.models.event import DeploymentEvent, EventType, Severity
from gram_deploy.models.action_item import ActionItem, ActionItemStatus, Priority
from gram_deploy.models.insight import DeploymentInsight, InsightType
from gram_deploy.models.timeline import TimelineSegment, TimeAlignment

__all__ = [
    "Deployment",
    "DeploymentStatus",
    "Source",
    "SourceFile",
    "DeviceType",
    "TranscriptStatus",
    "RawTranscript",
    "TranscriptSegment",
    "WordTiming",
    "Person",
    "VoiceSample",
    "SpeakerMapping",
    "ResolutionMethod",
    "CanonicalUtterance",
    "UtteranceSource",
    "DeploymentEvent",
    "EventType",
    "Severity",
    "ActionItem",
    "ActionItemStatus",
    "Priority",
    "DeploymentInsight",
    "InsightType",
    "TimelineSegment",
    "TimeAlignment",
]
