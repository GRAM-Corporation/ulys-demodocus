"""Tests for GRAM data models - validation and serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from gram_deploy.models import (
    # Base
    GRAMModel,
    utc_now,
    ensure_utc,
    # Deployment
    Deployment,
    DeploymentStatus,
    # Source
    Source,
    SourceFile,
    DeviceType,
    TranscriptStatus,
    # Transcript
    RawTranscript,
    TranscriptSegment,
    TranscriptSpeaker,
    WordTiming,
    # Person
    Person,
    VoiceSample,
    # Speaker Mapping
    SpeakerMapping,
    ResolutionMethod,
    # Canonical Utterance
    CanonicalUtterance,
    CanonicalWord,
    UtteranceSource,
    get_utterances_in_range,
    # Event
    DeploymentEvent,
    EventType,
    Severity,
    # Action Item
    ActionItem,
    ActionItemStatus,
    Priority,
    # Insight
    DeploymentInsight,
    InsightType,
    TimeRange,
    # Timeline
    TimelineSegment,
    TimeAlignment,
)


# =============================================================================
# Base Model Tests
# =============================================================================

class TestBaseModel:
    """Tests for GRAMModel base class."""

    def test_utc_now_returns_timezone_aware(self):
        """utc_now() should return timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_ensure_utc_naive_datetime(self):
        """ensure_utc() should add UTC timezone to naive datetime."""
        naive = datetime(2025, 1, 19, 12, 0, 0)
        result = ensure_utc(naive)
        assert result.tzinfo == timezone.utc
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 19

    def test_ensure_utc_none(self):
        """ensure_utc() should return None for None input."""
        assert ensure_utc(None) is None


# =============================================================================
# Deployment Tests
# =============================================================================

class TestDeployment:
    """Tests for Deployment model."""

    def test_valid_deployment(self):
        """Should create a valid deployment."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="Vinci",
            date="2025-01-19",
        )
        assert deployment.id == "deploy:20250119_vinci_01"
        assert deployment.location == "Vinci"
        assert deployment.status == DeploymentStatus.INGESTING

    def test_invalid_id_pattern(self):
        """Should reject invalid deployment ID."""
        with pytest.raises(ValueError):
            Deployment(
                id="invalid_id",
                location="Vinci",
                date="2025-01-19",
            )

    def test_invalid_date_format(self):
        """Should reject invalid date format."""
        with pytest.raises(ValueError):
            Deployment(
                id="deploy:20250119_vinci_01",
                location="Vinci",
                date="01-19-2025",  # Wrong format
            )

    def test_generate_id(self):
        """Should generate correct deployment ID."""
        deployment_id = Deployment.generate_id("Vinci", "2025-01-19", 1)
        assert deployment_id == "deploy:20250119_vinci_01"

    def test_generate_id_special_chars(self):
        """Should handle special characters in location."""
        deployment_id = Deployment.generate_id("New York City!", "2025-01-19", 2)
        assert deployment_id == "deploy:20250119_new_york_city_02"

    def test_canonical_time_range_validation(self):
        """Should reject end time before start time."""
        with pytest.raises(ValueError, match="canonical_start_time must be <= canonical_end_time"):
            Deployment(
                id="deploy:20250119_vinci_01",
                location="Vinci",
                date="2025-01-19",
                canonical_start_time=datetime(2025, 1, 19, 12, 0, 0),
                canonical_end_time=datetime(2025, 1, 19, 10, 0, 0),  # Before start
            )

    def test_duration_seconds(self):
        """Should calculate duration correctly."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="Vinci",
            date="2025-01-19",
            canonical_start_time=datetime(2025, 1, 19, 10, 0, 0),
            canonical_end_time=datetime(2025, 1, 19, 11, 30, 0),
        )
        assert deployment.duration_seconds == 5400.0  # 1.5 hours

    def test_get_source_path(self):
        """Should return correct source path."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="Vinci",
            date="2025-01-19",
        )
        source_id = "source:deploy:20250119_vinci_01/gopro_01"
        path = deployment.get_source_path(source_id)
        assert path == Path("deployments/deploy_20250119_vinci_01/sources/gopro_01")

    def test_get_source_path_wrong_deployment(self):
        """Should reject source from different deployment."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="Vinci",
            date="2025-01-19",
        )
        source_id = "source:deploy:20250120_other_01/gopro_01"
        with pytest.raises(ValueError, match="does not belong to deployment"):
            deployment.get_source_path(source_id)

    def test_serialization_roundtrip(self):
        """Should serialize and deserialize correctly."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="Vinci",
            date="2025-01-19",
            notes="Test deployment",
        )
        json_str = deployment.to_json()
        restored = Deployment.from_json(json_str)
        assert restored.id == deployment.id
        assert restored.location == deployment.location
        assert restored.notes == deployment.notes


# =============================================================================
# Source Tests
# =============================================================================

class TestSource:
    """Tests for Source model."""

    def test_valid_source(self):
        """Should create a valid source."""
        source = Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
        )
        assert source.device_type == DeviceType.GOPRO
        assert source.transcript_status == TranscriptStatus.PENDING

    def test_invalid_id_pattern(self):
        """Should reject invalid source ID."""
        with pytest.raises(ValueError):
            Source(
                id="invalid_source_id",
                deployment_id="deploy:20250119_vinci_01",
                device_type=DeviceType.GOPRO,
                device_number=1,
            )

    def test_generate_id(self):
        """Should generate correct source ID."""
        source_id = Source.generate_id(
            "deploy:20250119_vinci_01",
            DeviceType.GOPRO,
            1
        )
        assert source_id == "source:deploy:20250119_vinci_01/gopro_01"


class TestSourceFile:
    """Tests for SourceFile model."""

    def test_auto_compute_end_offset(self):
        """Should auto-compute end offset from duration."""
        file = SourceFile(
            filename="GX010006.MP4",
            file_path="/path/to/file.mp4",
            duration_seconds=60.0,
            start_offset_ms=1000,
        )
        assert file.end_offset_ms == 61000  # 1000 + 60000

    def test_time_range_validation(self):
        """Should reject invalid time range."""
        with pytest.raises(ValueError, match="end_offset_ms must be >= start_offset_ms"):
            SourceFile(
                filename="test.mp4",
                file_path="/path/to/file.mp4",
                duration_seconds=10.0,
                start_offset_ms=5000,
                end_offset_ms=1000,  # Before start
            )


# =============================================================================
# Transcript Tests
# =============================================================================

class TestTranscript:
    """Tests for transcript models."""

    def test_word_timing_time_range(self):
        """Should reject invalid word timing range."""
        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            WordTiming(
                text="hello",
                start_time=5.0,
                end_time=3.0,  # Before start
            )

    def test_transcript_segment_time_range(self):
        """Should reject invalid segment time range."""
        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            TranscriptSegment(
                text="Hello world",
                start_time=10.0,
                end_time=5.0,  # Before start
            )

    def test_transcript_segment_duration(self):
        """Should calculate duration correctly."""
        segment = TranscriptSegment(
            text="Hello world",
            start_time=5.0,
            end_time=10.0,
        )
        assert segment.duration_seconds == 5.0

    def test_raw_transcript_total_duration(self):
        """Should calculate total duration from segments."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(text="One", start_time=0.0, end_time=2.0),
                TranscriptSegment(text="Two", start_time=2.0, end_time=5.0),
                TranscriptSegment(text="Three", start_time=5.0, end_time=10.0),
            ],
        )
        assert transcript.total_duration == 10.0


# =============================================================================
# Person Tests
# =============================================================================

class TestPerson:
    """Tests for Person model."""

    def test_valid_person(self):
        """Should create a valid person."""
        person = Person(
            id="person:damion",
            name="Damion Shelton",
            role="CTO",
        )
        assert person.id == "person:damion"
        assert person.name == "Damion Shelton"

    def test_invalid_id_pattern(self):
        """Should reject invalid person ID."""
        with pytest.raises(ValueError):
            Person(
                id="damion",  # Missing "person:" prefix
                name="Damion Shelton",
            )

    def test_generate_id(self):
        """Should generate correct person ID."""
        person_id = Person.generate_id("Damion Shelton")
        assert person_id == "person:damion_shelton"

    def test_from_name(self):
        """Should create person from name."""
        person = Person.from_name("John Doe", role="Engineer")
        assert person.name == "John Doe"
        assert person.id == "person:john_doe"
        assert person.role == "Engineer"

    def test_matches_name(self):
        """Should match name variations."""
        person = Person(
            id="person:damion",
            name="Damion Shelton",
            aliases=["D", "Dam"],
        )
        assert person.matches_name("Damion Shelton")
        assert person.matches_name("Damion")  # First name
        assert person.matches_name("D")  # Alias
        assert person.matches_name("Dam")  # Alias
        assert not person.matches_name("John")


class TestVoiceSample:
    """Tests for VoiceSample model."""

    def test_time_range_validation(self):
        """Should reject invalid time range."""
        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            VoiceSample(
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                start_time=10.0,
                end_time=5.0,  # Before start
            )


# =============================================================================
# Canonical Utterance Tests
# =============================================================================

class TestCanonicalUtterance:
    """Tests for CanonicalUtterance model."""

    def test_valid_utterance(self):
        """Should create a valid utterance."""
        utterance = CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/abc123",
            deployment_id="deploy:20250119_vinci_01",
            text="Hello world",
            canonical_start_ms=0,
            canonical_end_ms=2000,
        )
        assert utterance.duration_ms == 2000
        assert utterance.duration_seconds == 2.0

    def test_time_range_validation(self):
        """Should reject invalid time range."""
        with pytest.raises(ValueError, match="canonical_start_ms must be <= canonical_end_ms"):
            CanonicalUtterance(
                id="utterance:deploy:20250119_vinci_01/abc123",
                deployment_id="deploy:20250119_vinci_01",
                text="Hello",
                canonical_start_ms=5000,
                canonical_end_ms=2000,  # Before start
            )

    def test_overlaps_with(self):
        """Should detect overlapping utterances."""
        u1 = CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/a",
            deployment_id="deploy:20250119_vinci_01",
            text="First",
            canonical_start_ms=0,
            canonical_end_ms=2000,
        )
        u2 = CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/b",
            deployment_id="deploy:20250119_vinci_01",
            text="Second",
            canonical_start_ms=1000,
            canonical_end_ms=3000,
        )
        u3 = CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/c",
            deployment_id="deploy:20250119_vinci_01",
            text="Third",
            canonical_start_ms=5000,
            canonical_end_ms=7000,
        )
        assert u1.overlaps_with(u2)  # 50% overlap
        assert not u1.overlaps_with(u3)  # No overlap

    def test_is_in_range(self):
        """Should detect if utterance is in time range."""
        utterance = CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/a",
            deployment_id="deploy:20250119_vinci_01",
            text="Test",
            canonical_start_ms=1000,
            canonical_end_ms=3000,
        )
        assert utterance.is_in_range(0, 2000)  # Partial overlap
        assert utterance.is_in_range(2000, 5000)  # Partial overlap
        assert utterance.is_in_range(500, 3500)  # Contains
        assert not utterance.is_in_range(5000, 7000)  # No overlap


class TestGetUtterancesInRange:
    """Tests for get_utterances_in_range function."""

    @pytest.fixture
    def utterances(self):
        """Create test utterances."""
        return [
            CanonicalUtterance(
                id="utterance:deploy:20250119_vinci_01/a",
                deployment_id="deploy:20250119_vinci_01",
                text="First",
                canonical_start_ms=0,
                canonical_end_ms=2000,
            ),
            CanonicalUtterance(
                id="utterance:deploy:20250119_vinci_01/b",
                deployment_id="deploy:20250119_vinci_01",
                text="Second",
                canonical_start_ms=3000,
                canonical_end_ms=5000,
            ),
            CanonicalUtterance(
                id="utterance:deploy:20250119_vinci_01/c",
                deployment_id="deploy:20250119_vinci_01",
                text="Third",
                canonical_start_ms=6000,
                canonical_end_ms=8000,
            ),
        ]

    def test_overlapping_range(self, utterances):
        """Should return utterances overlapping the range."""
        result = get_utterances_in_range(utterances, 1000, 4000)
        assert len(result) == 2
        assert result[0].text == "First"
        assert result[1].text == "Second"

    def test_fully_contained(self, utterances):
        """Should return only fully contained utterances."""
        result = get_utterances_in_range(utterances, 0, 10000, fully_contained=True)
        assert len(result) == 3

        result = get_utterances_in_range(utterances, 1000, 7000, fully_contained=True)
        assert len(result) == 1
        assert result[0].text == "Second"

    def test_empty_range(self, utterances):
        """Should return empty for non-overlapping range."""
        result = get_utterances_in_range(utterances, 10000, 12000)
        assert len(result) == 0


# =============================================================================
# Timeline Tests
# =============================================================================

class TestTimeRange:
    """Tests for TimeRange model."""

    def test_valid_range(self):
        """Should create valid time range."""
        tr = TimeRange(start_ms=1000, end_ms=5000)
        assert tr.duration_ms == 4000

    def test_invalid_range(self):
        """Should reject invalid range."""
        with pytest.raises(ValueError, match="start_ms must be <= end_ms"):
            TimeRange(start_ms=5000, end_ms=1000)


class TestTimelineSegment:
    """Tests for TimelineSegment model."""

    def test_valid_segment(self):
        """Should create valid segment."""
        segment = TimelineSegment(
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            canonical_start_ms=0,
            canonical_end_ms=5000,
            file_index=0,
        )
        assert segment.duration_ms == 5000

    def test_invalid_time_range(self):
        """Should reject invalid time range."""
        with pytest.raises(ValueError, match="canonical_start_ms must be <= canonical_end_ms"):
            TimelineSegment(
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                canonical_start_ms=5000,
                canonical_end_ms=1000,  # Before start
                file_index=0,
            )


# =============================================================================
# Serialization Tests with Sample Data
# =============================================================================

class TestSampleDataLoading:
    """Tests for loading sample data files."""

    @pytest.fixture
    def sample_data_dir(self):
        """Get sample data directory."""
        return Path(__file__).parent.parent / "deployments"

    def test_load_deployment_json(self, sample_data_dir):
        """Should load deployment.json correctly."""
        deployment_file = sample_data_dir / "deploy_20250119_vinci_01" / "deployment.json"
        if not deployment_file.exists():
            pytest.skip("Sample deployment.json not found")

        deployment = Deployment.load_from_file(deployment_file)
        assert deployment.id == "deploy:20250119_vinci_01"
        assert deployment.location == "Vinci"
        assert deployment.date == "2025-01-19"
        assert "source:deploy:20250119_vinci_01/gopro_01" in deployment.sources

    def test_load_source_json(self, sample_data_dir):
        """Should load source.json correctly."""
        source_file = sample_data_dir / "deploy_20250119_vinci_01" / "sources" / "gopro_01" / "source.json"
        if not source_file.exists():
            pytest.skip("Sample source.json not found")

        source = Source.load_from_file(source_file)
        assert source.id == "source:deploy:20250119_vinci_01/gopro_01"
        assert source.device_type == DeviceType.GOPRO
        assert len(source.files) > 0

    def test_load_raw_transcript_json(self, sample_data_dir):
        """Should load raw_transcript.json correctly."""
        transcript_file = sample_data_dir / "deploy_20250119_vinci_01" / "sources" / "gopro_01" / "raw_transcript.json"
        if not transcript_file.exists():
            pytest.skip("Sample raw_transcript.json not found")

        # The sample data doesn't have the full transcript structure, just the inner data
        with open(transcript_file) as f:
            data = json.load(f)

        # Create RawTranscript from the data
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            language_code=data.get("language_code", "en"),
            segments=[
                TranscriptSegment(
                    text=seg["text"],
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    speaker=TranscriptSpeaker(**seg["speaker"]) if seg.get("speaker") else None,
                    words=[WordTiming(**w) for w in seg.get("words", [])] if seg.get("words") else None,
                )
                for seg in data.get("segments", [])
            ],
        )
        assert len(transcript.segments) > 0
        assert transcript.total_duration > 0


# =============================================================================
# File I/O Tests
# =============================================================================

class TestFileIO:
    """Tests for file save/load operations."""

    def test_save_and_load_deployment(self):
        """Should save and load deployment from file."""
        deployment = Deployment(
            id="deploy:20250119_test_01",
            location="Test Location",
            date="2025-01-19",
            notes="Test notes",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "deployment.json"
            deployment.save_to_file(file_path)

            loaded = Deployment.load_from_file(file_path)
            assert loaded.id == deployment.id
            assert loaded.location == deployment.location
            assert loaded.notes == deployment.notes

    def test_save_creates_parent_dirs(self):
        """Should create parent directories when saving."""
        deployment = Deployment(
            id="deploy:20250119_test_01",
            location="Test",
            date="2025-01-19",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nested" / "deep" / "deployment.json"
            deployment.save_to_file(file_path)
            assert file_path.exists()
