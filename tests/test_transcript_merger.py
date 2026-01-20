"""Tests for the TranscriptMerger service."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.deployment import Deployment
from gram_deploy.models.source import Source, SourceFile, DeviceType
from gram_deploy.models.transcript import (
    RawTranscript,
    TranscriptSegment,
    TranscriptSpeaker,
    WordTiming,
)
from gram_deploy.models.speaker_mapping import SpeakerMapping, ResolutionMethod
from gram_deploy.models.canonical_utterance import CanonicalUtterance, UtteranceSource

# Import transcript_merger directly to avoid services/__init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "transcript_merger",
    Path(__file__).parent.parent
    / "src"
    / "gram_deploy"
    / "services"
    / "transcript_merger.py",
)
tm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm_module)
TranscriptMerger = tm_module.TranscriptMerger


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def merger(temp_data_dir):
    """Create a TranscriptMerger with a temp data directory."""
    return TranscriptMerger(temp_data_dir)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
        sources=[
            "source:deploy:20250119_vinci_01/gopro_01",
            "source:deploy:20250119_vinci_01/gopro_02",
        ],
    )


@pytest.fixture
def sample_source_gopro_01():
    """Create a sample source for gopro_01."""
    return Source(
        id="source:deploy:20250119_vinci_01/gopro_01",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=1,
        canonical_offset_ms=0,  # Reference source
        files=[
            SourceFile(
                filename="GX010001.MP4",
                file_path="/path/to/GX010001.MP4",
                duration_seconds=600.0,
                start_offset_ms=0,
            )
        ],
        total_duration_seconds=600.0,
    )


@pytest.fixture
def sample_source_gopro_02():
    """Create a sample source for gopro_02 with offset."""
    return Source(
        id="source:deploy:20250119_vinci_01/gopro_02",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=2,
        canonical_offset_ms=5000,  # Starts 5 seconds after reference
        files=[
            SourceFile(
                filename="GX020001.MP4",
                file_path="/path/to/GX020001.MP4",
                duration_seconds=600.0,
                start_offset_ms=0,
            )
        ],
        total_duration_seconds=600.0,
    )


@pytest.fixture
def sample_transcript_a():
    """Create a sample transcript for source A."""
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_01",
        source_id="source:deploy:20250119_vinci_01/gopro_01",
        transcription_service="elevenlabs",
        segments=[
            TranscriptSegment(
                text="We need to check the Starlink battery status",
                start_time=10.0,
                end_time=13.0,
                speaker=TranscriptSpeaker(id="speaker_A"),
                confidence=0.95,
            ),
            TranscriptSegment(
                text="The signal strength looks good from here",
                start_time=20.0,
                end_time=23.0,
                speaker=TranscriptSpeaker(id="speaker_B"),
                confidence=0.92,
            ),
            TranscriptSegment(
                text="Let me take some measurements over there",
                start_time=30.0,
                end_time=33.0,
                speaker=TranscriptSpeaker(id="speaker_A"),
                confidence=0.88,
            ),
        ],
    )


@pytest.fixture
def sample_transcript_b():
    """Create a sample transcript for source B with matching content."""
    # Source B starts 5 seconds after A but has same speech
    # So in source B's local time, the same phrases appear 5 seconds earlier
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_02",
        source_id="source:deploy:20250119_vinci_01/gopro_02",
        transcription_service="elevenlabs",
        segments=[
            # This should match A's first segment when aligned
            TranscriptSegment(
                text="We need to check the Starlink battery status",
                start_time=5.0,  # Local time in B = 5s, canonical = 5 + 5 = 10
                end_time=8.0,
                speaker=TranscriptSpeaker(id="speaker_1"),  # Different raw ID
                confidence=0.90,
            ),
            TranscriptSegment(
                text="The signal strength looks good from here",
                start_time=15.0,  # Local = 15s, canonical = 15 + 5 = 20
                end_time=18.0,
                speaker=TranscriptSpeaker(id="speaker_2"),
                confidence=0.85,
            ),
        ],
    )


@pytest.fixture
def sample_speaker_mappings():
    """Create speaker mappings for testing."""
    return {
        "source:deploy:20250119_vinci_01/gopro_01": [
            SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.9,
                method=ResolutionMethod.VOICE_MATCH,
            ),
            SpeakerMapping(
                raw_speaker_id="speaker_B",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:john",
                confidence=0.85,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ],
        "source:deploy:20250119_vinci_01/gopro_02": [
            SpeakerMapping(
                raw_speaker_id="speaker_1",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_02",
                resolved_person_id="person:damion",
                confidence=0.8,
                method=ResolutionMethod.VOICE_MATCH,
            ),
            SpeakerMapping(
                raw_speaker_id="speaker_2",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_02",
                resolved_person_id="person:john",
                confidence=0.75,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ],
    }


class TestTimeConversion:
    """Tests for _convert_to_canonical_time."""

    def test_convert_to_canonical_time_no_offset(
        self, merger, sample_source_gopro_01, sample_deployment
    ):
        """Test time conversion with zero offset."""
        segment = TranscriptSegment(
            text="Test text",
            start_time=10.0,
            end_time=15.0,
            speaker=TranscriptSpeaker(id="speaker_A"),
        )

        result = merger._convert_to_canonical_time(
            segment=segment,
            segment_index=0,
            source=sample_source_gopro_01,
            deployment=sample_deployment,
            mapping_lookup={},
        )

        assert result.canonical_start_ms == 10000
        assert result.canonical_end_ms == 15000
        assert result.text == "Test text"

    def test_convert_to_canonical_time_with_offset(
        self, merger, sample_source_gopro_02, sample_deployment
    ):
        """Test time conversion with 5 second offset."""
        segment = TranscriptSegment(
            text="Test text",
            start_time=10.0,
            end_time=15.0,
            speaker=TranscriptSpeaker(id="speaker_1"),
        )

        result = merger._convert_to_canonical_time(
            segment=segment,
            segment_index=0,
            source=sample_source_gopro_02,  # Has 5000ms offset
            deployment=sample_deployment,
            mapping_lookup={},
        )

        # 10s local + 5s offset = 15s canonical
        assert result.canonical_start_ms == 15000
        assert result.canonical_end_ms == 20000

    def test_convert_to_canonical_time_with_words(
        self, merger, sample_source_gopro_01, sample_deployment
    ):
        """Test time conversion includes word timings."""
        segment = TranscriptSegment(
            text="Hello world",
            start_time=10.0,
            end_time=12.0,
            words=[
                WordTiming(text="Hello", start_time=10.0, end_time=10.5),
                WordTiming(text="world", start_time=10.6, end_time=12.0),
            ],
        )

        result = merger._convert_to_canonical_time(
            segment=segment,
            segment_index=0,
            source=sample_source_gopro_01,
            deployment=sample_deployment,
            mapping_lookup={},
        )

        assert result.words is not None
        assert len(result.words) == 2
        assert result.words[0].text == "Hello"
        assert result.words[0].canonical_start_ms == 10000
        assert result.words[0].canonical_end_ms == 10500

    def test_convert_to_canonical_time_with_speaker_mapping(
        self, merger, sample_source_gopro_01, sample_deployment
    ):
        """Test speaker resolution during conversion."""
        segment = TranscriptSegment(
            text="Test text",
            start_time=10.0,
            end_time=15.0,
            speaker=TranscriptSpeaker(id="speaker_A"),
        )

        mapping_lookup = {
            "source:deploy:20250119_vinci_01/gopro_01/speaker_A": SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.9,
                method=ResolutionMethod.VOICE_MATCH,
            )
        }

        result = merger._convert_to_canonical_time(
            segment=segment,
            segment_index=0,
            source=sample_source_gopro_01,
            deployment=sample_deployment,
            mapping_lookup=mapping_lookup,
        )

        assert result.speaker_id == "person:damion"
        assert result.speaker_confidence == 0.9

    def test_convert_to_canonical_time_unmapped_speaker(
        self, merger, sample_source_gopro_01, sample_deployment
    ):
        """Test unmapped speakers become unknown_N."""
        segment = TranscriptSegment(
            text="Test text",
            start_time=10.0,
            end_time=15.0,
            speaker=TranscriptSpeaker(id="speaker_X"),
        )

        result = merger._convert_to_canonical_time(
            segment=segment,
            segment_index=0,
            source=sample_source_gopro_01,
            deployment=sample_deployment,
            mapping_lookup={},  # No mappings
        )

        assert result.speaker_id == "unknown_speaker_X"
        assert result.speaker_confidence == 0.0


class TestDuplicateDetection:
    """Tests for _detect_duplicates."""

    def test_detect_duplicates_same_text_same_time(self, merger, sample_deployment):
        """Test detection of identical duplicates."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.8,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=5.0,
                        local_end_time=8.0,
                    )
                ],
            ),
        ]

        groups = merger._detect_duplicates(utterances)

        assert len(groups) == 1
        assert groups[0] == [0, 1]

    def test_detect_duplicates_similar_text(self, merger, sample_deployment):
        """Test detection of slightly different text as duplicates."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="We need to check the battery status",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="We need to check the battery status now",  # Slightly different
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.8,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=5.0,
                        local_end_time=8.0,
                    )
                ],
            ),
        ]

        groups = merger._detect_duplicates(utterances)

        # Should detect as duplicates (text similarity > 0.85)
        assert len(groups) == 1

    def test_detect_duplicates_different_speakers(self, merger, sample_deployment):
        """Test non-detection when speakers differ."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:john",  # Different speaker
                speaker_confidence=0.8,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=5.0,
                        local_end_time=8.0,
                    )
                ],
            ),
        ]

        groups = merger._detect_duplicates(utterances)

        # Different speakers - not duplicates
        assert len(groups) == 0

    def test_detect_duplicates_unknown_speaker_matches(self, merger, sample_deployment):
        """Test that unknown speakers can match resolved speakers."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="unknown_speaker_X",  # Unknown can match anyone
                speaker_confidence=0.0,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=5.0,
                        local_end_time=8.0,
                    )
                ],
            ),
        ]

        groups = merger._detect_duplicates(utterances)

        # Unknown speaker should match resolved speaker
        assert len(groups) == 1

    def test_detect_duplicates_no_overlap(self, merger, sample_deployment):
        """Test non-detection when times don't overlap."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=50000,  # Way later
                canonical_end_ms=53000,
                speaker_id="person:damion",
                speaker_confidence=0.8,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=45.0,
                        local_end_time=48.0,
                    )
                ],
            ),
        ]

        groups = merger._detect_duplicates(utterances)

        # Times too far apart
        assert len(groups) == 0


class TestConflictResolution:
    """Tests for _resolve_conflicts."""

    def test_resolve_conflicts_picks_higher_confidence(self, merger, sample_deployment):
        """Test that higher confidence utterance is kept."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.5,  # Lower
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="This is a test",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,  # Higher
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=5.0,
                        local_end_time=8.0,
                    )
                ],
            ),
        ]

        duplicate_groups = [[0, 1]]
        result = merger._resolve_conflicts(utterances, duplicate_groups)

        assert len(result) == 1
        assert result[0].speaker_confidence == 0.9  # Higher confidence kept
        assert result[0].is_duplicate is True
        assert len(result[0].sources) == 2  # Both sources preserved

    def test_resolve_conflicts_picks_longer_text(self, merger, sample_deployment):
        """Test that more complete text is preferred."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="This is a test",  # Shorter
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="This is a test of the complete transcription",  # Longer
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,  # Same confidence
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_02",
                        local_start_time=5.0,
                        local_end_time=8.0,
                    )
                ],
            ),
        ]

        duplicate_groups = [[0, 1]]
        result = merger._resolve_conflicts(utterances, duplicate_groups)

        assert len(result) == 1
        # Longer text should be kept
        assert "complete transcription" in result[0].text


class TestTextSimilarity:
    """Tests for _text_similarity."""

    def test_text_similarity_identical(self, merger):
        """Test similarity of identical strings."""
        similarity = merger._text_similarity(
            "This is a test", "This is a test"
        )
        assert similarity == 1.0

    def test_text_similarity_case_insensitive(self, merger):
        """Test that comparison is case-insensitive."""
        similarity = merger._text_similarity(
            "THIS IS A TEST", "this is a test"
        )
        assert similarity == 1.0

    def test_text_similarity_high(self, merger):
        """Test high similarity for nearly identical strings."""
        similarity = merger._text_similarity(
            "We need to check the battery status",
            "We need to check the battery status now",
        )
        assert similarity > 0.85

    def test_text_similarity_low(self, merger):
        """Test low similarity for different strings."""
        similarity = merger._text_similarity(
            "The weather is nice today",
            "Check the battery status",
        )
        assert similarity < 0.5

    def test_text_similarity_empty(self, merger):
        """Test similarity with empty strings."""
        assert merger._text_similarity("", "test") == 0.0
        assert merger._text_similarity("test", "") == 0.0
        assert merger._text_similarity("", "") == 0.0


class TestMergeTranscripts:
    """Tests for the main merge_transcripts method."""

    def test_merge_transcripts_single_source(
        self,
        merger,
        sample_deployment,
        sample_source_gopro_01,
        sample_transcript_a,
    ):
        """Test merging with a single source."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=["source:deploy:20250119_vinci_01/gopro_01"],
        )

        result = merger.merge_transcripts(
            deployment=deployment,
            sources=[sample_source_gopro_01],
            transcripts=[sample_transcript_a],
            speaker_mappings={},
        )

        # All 3 segments should be converted
        assert len(result) == 3

        # First utterance should have correct canonical times
        assert result[0].canonical_start_ms == 10000
        assert result[0].canonical_end_ms == 13000

        # Should be sorted by time
        assert result[0].canonical_start_ms < result[1].canonical_start_ms
        assert result[1].canonical_start_ms < result[2].canonical_start_ms

    def test_merge_transcripts_multiple_sources_with_duplicates(
        self,
        merger,
        sample_deployment,
        sample_source_gopro_01,
        sample_source_gopro_02,
        sample_transcript_a,
        sample_transcript_b,
        sample_speaker_mappings,
    ):
        """Test merging multiple sources with duplicate detection."""
        result = merger.merge_transcripts(
            deployment=sample_deployment,
            sources=[sample_source_gopro_01, sample_source_gopro_02],
            transcripts=[sample_transcript_a, sample_transcript_b],
            speaker_mappings=sample_speaker_mappings,
        )

        # Source A has 3 segments, Source B has 2 segments
        # 2 should be detected as duplicates
        # So we should have 3 unique segments (with duplicates merged)
        assert len(result) <= 5  # At most all segments
        assert len(result) >= 3  # At least unique ones

        # Check that duplicates have multiple sources
        duplicate_utterances = [u for u in result if u.is_duplicate]
        assert len(duplicate_utterances) > 0

        # Duplicates should have sources from both cameras
        for dup in duplicate_utterances:
            source_ids = {s.source_id for s in dup.sources}
            assert len(source_ids) == 2

    def test_merge_transcripts_speaker_resolution(
        self,
        merger,
        sample_deployment,
        sample_source_gopro_01,
        sample_transcript_a,
        sample_speaker_mappings,
    ):
        """Test that speaker mappings are applied correctly."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=["source:deploy:20250119_vinci_01/gopro_01"],
        )

        result = merger.merge_transcripts(
            deployment=deployment,
            sources=[sample_source_gopro_01],
            transcripts=[sample_transcript_a],
            speaker_mappings=sample_speaker_mappings,
        )

        # Check speaker IDs are resolved
        speakers = {u.speaker_id for u in result}
        assert "person:damion" in speakers
        assert "person:john" in speakers

    def test_merge_transcripts_empty_input(self, merger, sample_deployment):
        """Test merging with no sources."""
        result = merger.merge_transcripts(
            deployment=sample_deployment,
            sources=[],
            transcripts=[],
            speaker_mappings={},
        )

        assert result == []


class TestSaveLoadCanonicalTranscript:
    """Tests for saving and loading canonical transcripts."""

    def test_save_and_load_canonical_transcript(
        self, merger, temp_data_dir, sample_deployment
    ):
        """Test saving and loading canonical transcript."""
        # Create test utterances
        utterances = [
            CanonicalUtterance(
                id="utterance:deploy:20250119_vinci_01/001",
                deployment_id=sample_deployment.id,
                text="Test utterance one",
                canonical_start_ms=10000,
                canonical_end_ms=13000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=10.0,
                        local_end_time=13.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:deploy:20250119_vinci_01/002",
                deployment_id=sample_deployment.id,
                text="Test utterance two",
                canonical_start_ms=20000,
                canonical_end_ms=23000,
                speaker_id="person:john",
                speaker_confidence=0.85,
                sources=[
                    UtteranceSource(
                        source_id="source:deploy:20250119_vinci_01/gopro_01",
                        local_start_time=20.0,
                        local_end_time=23.0,
                    )
                ],
            ),
        ]

        # Save
        save_path = merger.save_canonical_transcript(sample_deployment, utterances)

        assert save_path.exists()
        assert save_path.name == "canonical_transcript.json"

        # Load
        loaded = merger.load_canonical_transcript(sample_deployment)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].text == "Test utterance one"
        assert loaded[1].text == "Test utterance two"
        assert loaded[0].speaker_id == "person:damion"

    def test_load_nonexistent_transcript(self, merger, sample_deployment):
        """Test loading when no transcript exists."""
        result = merger.load_canonical_transcript(sample_deployment)
        assert result is None


class TestMergeAdjacent:
    """Tests for merging adjacent segments from same speaker."""

    def test_merge_adjacent_same_speaker(self, merger, sample_deployment):
        """Test merging adjacent segments from same speaker."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="First part",
                canonical_start_ms=10000,
                canonical_end_ms=12000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=10.0,
                        local_end_time=12.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="second part",
                canonical_start_ms=12500,  # Small gap (500ms)
                canonical_end_ms=14000,
                speaker_id="person:damion",  # Same speaker
                speaker_confidence=0.85,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=12.5,
                        local_end_time=14.0,
                    )
                ],
            ),
        ]

        result = merger._merge_adjacent(utterances)

        # Should merge into one utterance
        assert len(result) == 1
        assert "First part second part" in result[0].text
        assert result[0].canonical_start_ms == 10000
        assert result[0].canonical_end_ms == 14000

    def test_merge_adjacent_different_speakers(self, merger, sample_deployment):
        """Test that different speakers are not merged."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="First part",
                canonical_start_ms=10000,
                canonical_end_ms=12000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=10.0,
                        local_end_time=12.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="second part",
                canonical_start_ms=12500,
                canonical_end_ms=14000,
                speaker_id="person:john",  # Different speaker
                speaker_confidence=0.85,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=12.5,
                        local_end_time=14.0,
                    )
                ],
            ),
        ]

        result = merger._merge_adjacent(utterances)

        # Should remain separate
        assert len(result) == 2

    def test_merge_adjacent_large_gap(self, merger, sample_deployment):
        """Test that large gaps prevent merging."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="First part",
                canonical_start_ms=10000,
                canonical_end_ms=12000,
                speaker_id="person:damion",
                speaker_confidence=0.9,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=10.0,
                        local_end_time=12.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="second part",
                canonical_start_ms=20000,  # Large gap (8 seconds)
                canonical_end_ms=22000,
                speaker_id="person:damion",  # Same speaker
                speaker_confidence=0.85,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=20.0,
                        local_end_time=22.0,
                    )
                ],
            ),
        ]

        result = merger._merge_adjacent(utterances)

        # Gap too large - should remain separate
        assert len(result) == 2

    def test_merge_adjacent_unknown_speaker_not_merged(self, merger, sample_deployment):
        """Test that unknown speakers are not merged."""
        utterances = [
            CanonicalUtterance(
                id="utterance:test/001",
                deployment_id=sample_deployment.id,
                text="First part",
                canonical_start_ms=10000,
                canonical_end_ms=12000,
                speaker_id="unknown_A",
                speaker_confidence=0.0,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=10.0,
                        local_end_time=12.0,
                    )
                ],
            ),
            CanonicalUtterance(
                id="utterance:test/002",
                deployment_id=sample_deployment.id,
                text="second part",
                canonical_start_ms=12500,
                canonical_end_ms=14000,
                speaker_id="unknown_A",  # Same unknown
                speaker_confidence=0.0,
                sources=[
                    UtteranceSource(
                        source_id="source:test",
                        local_start_time=12.5,
                        local_end_time=14.0,
                    )
                ],
            ),
        ]

        result = merger._merge_adjacent(utterances)

        # Unknown speakers should not be merged (they might be different people)
        assert len(result) == 2


class TestLegacyMergeMethod:
    """Tests for the legacy merge() method."""

    def test_legacy_merge_method(
        self,
        merger,
        sample_source_gopro_01,
        sample_transcript_a,
    ):
        """Test backward compatibility of legacy merge method."""
        mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.9,
                method=ResolutionMethod.VOICE_MATCH,
            ),
        ]

        result = merger.merge(
            sources=[sample_source_gopro_01],
            transcripts=[sample_transcript_a],
            speaker_mappings=mappings,
        )

        assert len(result) > 0
        # Check that utterances were created
        assert all(isinstance(u, CanonicalUtterance) for u in result)
