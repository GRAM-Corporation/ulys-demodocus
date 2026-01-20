"""Raw transcript entity - unprocessed output from transcription service."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    """Word-level timing information."""

    text: str
    start_time: float = Field(..., description="Start time in seconds from source beginning")
    end_time: float = Field(..., description="End time in seconds from source beginning")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class TranscriptSpeaker(BaseModel):
    """Speaker identification from transcription service."""

    id: str = Field(..., description="Service-assigned speaker ID")
    name: Optional[str] = Field(None, description="Service-assigned speaker name if available")


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech."""

    text: str
    start_time: float = Field(..., description="Start time in seconds from source beginning")
    end_time: float = Field(..., description="End time in seconds from source beginning")
    speaker: Optional[TranscriptSpeaker] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    words: Optional[list[WordTiming]] = None

    @property
    def duration_seconds(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_time - self.start_time


class RawTranscript(BaseModel):
    """Unprocessed transcript from a transcription service.

    Preserves the original data before merging or speaker resolution.

    ID format: transcript:source:{source_id}
    """

    id: str = Field(..., pattern=r"^transcript:source:.+$")
    source_id: str
    language_code: str = Field(default="en")
    transcription_service: str = Field(
        ..., description="Service used: elevenlabs, whisper, assemblyai, deepgram, other"
    )
    transcription_model: Optional[str] = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata from the transcription response
    audio_duration_seconds: Optional[float] = None
    word_count: Optional[int] = None

    @classmethod
    def generate_id(cls, source_id: str) -> str:
        """Generate transcript ID from source ID."""
        return f"transcript:{source_id}"

    @property
    def total_duration(self) -> float:
        """Total duration covered by transcript segments."""
        if not self.segments:
            return 0.0
        return max(s.end_time for s in self.segments)

    @property
    def speaker_ids(self) -> set[str]:
        """Unique speaker IDs in this transcript."""
        return {s.speaker.id for s in self.segments if s.speaker}

    def get_text(self, start_time: float = 0, end_time: Optional[float] = None) -> str:
        """Get concatenated text for a time range."""
        segments = self.segments
        if end_time is not None:
            segments = [s for s in segments if s.start_time >= start_time and s.end_time <= end_time]
        else:
            segments = [s for s in segments if s.start_time >= start_time]
        return " ".join(s.text for s in segments)
