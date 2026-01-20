"""Tests for the TimeAlignmentService."""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.source import Source, SourceFile, DeviceType, TranscriptStatus
from gram_deploy.models.transcript import RawTranscript, TranscriptSegment

# Import time_alignment directly to avoid services/__init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "time_alignment",
    Path(__file__).parent.parent
    / "src"
    / "gram_deploy"
    / "services"
    / "time_alignment.py",
)
ta_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta_module)
TimeAlignmentService = ta_module.TimeAlignmentService
AlignmentResult = ta_module.AlignmentResult
Match = ta_module.Match


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def service(temp_data_dir):
    """Create a TimeAlignmentService with a temp data directory."""
    return TimeAlignmentService(temp_data_dir)


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
    """Create a sample source for gopro_02."""
    return Source(
        id="source:deploy:20250119_vinci_01/gopro_02",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=2,
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
    """Create a sample transcript for source A (reference)."""
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_01",
        source_id="source:deploy:20250119_vinci_01/gopro_01",
        transcription_service="elevenlabs",
        segments=[
            TranscriptSegment(
                text="We need to check the Starlink battery status",
                start_time=10.0,
                end_time=13.0,
            ),
            TranscriptSegment(
                text="The signal strength looks good from here",
                start_time=20.0,
                end_time=23.0,
            ),
            TranscriptSegment(
                text="Let me take some measurements over there",
                start_time=30.0,
                end_time=33.0,
            ),
            TranscriptSegment(
                text="This is exactly what we were looking for",
                start_time=45.0,
                end_time=48.0,
            ),
            TranscriptSegment(
                text="Can you confirm the readings on your end?",
                start_time=60.0,
                end_time=63.0,
            ),
        ],
    )


@pytest.fixture
def sample_transcript_b():
    """Create a sample transcript for source B with known offset."""
    # Source B starts 5 seconds AFTER source A
    # So same phrases appear 5 seconds earlier in B's local time
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_02",
        source_id="source:deploy:20250119_vinci_01/gopro_02",
        transcription_service="elevenlabs",
        segments=[
            TranscriptSegment(
                text="We need to check the Starlink battery status",
                start_time=5.0,  # 10.0 - 5.0 = 5.0
                end_time=8.0,
            ),
            TranscriptSegment(
                text="The signal strength looks good from here",
                start_time=15.0,  # 20.0 - 5.0 = 15.0
                end_time=18.0,
            ),
            TranscriptSegment(
                text="Let me take some measurements over there",
                start_time=25.0,  # 30.0 - 5.0 = 25.0
                end_time=28.0,
            ),
            TranscriptSegment(
                text="This is exactly what we were looking for",
                start_time=40.0,  # 45.0 - 5.0 = 40.0
                end_time=43.0,
            ),
            TranscriptSegment(
                text="Can you confirm the readings on your end?",
                start_time=55.0,  # 60.0 - 5.0 = 55.0
                end_time=58.0,
            ),
        ],
    )


class TestAlignmentResultModel:
    """Tests for the AlignmentResult model."""

    def test_alignment_result_creation(self):
        """Test creating an AlignmentResult."""
        result = AlignmentResult(
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            canonical_offset_ms=5000,
            confidence=0.85,
            method="transcript",
            match_count=5,
        )

        assert result.source_id == "source:deploy:20250119_vinci_01/gopro_01"
        assert result.canonical_offset_ms == 5000
        assert result.confidence == 0.85
        assert result.method == "transcript"
        assert result.match_count == 5

    def test_alignment_result_confidence_bounds(self):
        """Test that confidence is bounded between 0 and 1."""
        with pytest.raises(ValueError):
            AlignmentResult(
                source_id="test",
                canonical_offset_ms=0,
                confidence=1.5,  # Invalid
                method="transcript",
            )

        with pytest.raises(ValueError):
            AlignmentResult(
                source_id="test",
                canonical_offset_ms=0,
                confidence=-0.1,  # Invalid
                method="transcript",
            )


class TestFuzzyMatching:
    """Tests for fuzzy matching functionality."""

    def test_fuzzy_ratio_identical_strings(self, service):
        """Test fuzzy ratio for identical strings."""
        text = "This is a test phrase"
        ratio = service._fuzzy_ratio(text, text)
        assert ratio == 1.0

    def test_fuzzy_ratio_completely_different(self, service):
        """Test fuzzy ratio for completely different strings."""
        ratio = service._fuzzy_ratio("apple orange banana", "xyz qwerty asdf")
        assert ratio < 0.3

    def test_fuzzy_ratio_high_similarity(self, service):
        """Test fuzzy ratio for highly similar strings."""
        # Minor differences - should be high similarity
        ratio = service._fuzzy_ratio(
            "We need to check the battery status",
            "We need to check the battery status now",
        )
        assert ratio > 0.8

    def test_fuzzy_ratio_case_insensitive(self, service):
        """Test that fuzzy matching is case insensitive."""
        ratio = service._fuzzy_ratio(
            "THIS IS A TEST", "this is a test"
        )
        assert ratio == 1.0

    def test_fuzzy_ratio_empty_strings(self, service):
        """Test fuzzy ratio with empty strings."""
        assert service._fuzzy_ratio("", "") == 0.0
        assert service._fuzzy_ratio("test", "") == 0.0
        assert service._fuzzy_ratio("", "test") == 0.0


class TestFindMatches:
    """Tests for finding matches between transcripts."""

    def test_find_matches_identical_transcripts(
        self, service, sample_transcript_a
    ):
        """Test finding matches with identical transcripts."""
        matches = service._find_matches(sample_transcript_a, sample_transcript_a)

        # Should find all 5 segments matching
        assert len(matches) == 5

        # All offsets should be 0 (same times)
        for match in matches:
            assert match.offset_ms == 0
            assert match.similarity > 0.85

    def test_find_matches_with_offset(
        self, service, sample_transcript_a, sample_transcript_b
    ):
        """Test finding matches with known offset."""
        matches = service._find_matches(sample_transcript_a, sample_transcript_b)

        # Should find all 5 matching segments
        assert len(matches) == 5

        # All offsets should be approximately 5000ms
        # (because B starts 5 seconds after A)
        for match in matches:
            assert 4500 <= match.offset_ms <= 5500
            assert match.similarity > 0.85

    def test_find_matches_no_overlap(self, service):
        """Test finding matches when transcripts don't overlap."""
        transcript_a = RawTranscript(
            id="transcript:source:test_a",
            source_id="source:test_a",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="This is completely different content",
                    start_time=10.0,
                    end_time=13.0,
                ),
            ],
        )
        transcript_b = RawTranscript(
            id="transcript:source:test_b",
            source_id="source:test_b",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="The weather is nice today",
                    start_time=10.0,
                    end_time=13.0,
                ),
            ],
        )

        matches = service._find_matches(transcript_a, transcript_b)
        assert len(matches) == 0


class TestOffsetClustering:
    """Tests for offset clustering algorithm."""

    def test_cluster_offsets_single_cluster(self, service):
        """Test clustering with all offsets in one cluster."""
        offsets = [5000, 5100, 4900, 5050, 4950]
        clusters = service._cluster_offsets(offsets, tolerance_ms=2000)

        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_cluster_offsets_multiple_clusters(self, service):
        """Test clustering with distinct clusters."""
        offsets = [1000, 1100, 5000, 5100, 10000, 10100]
        clusters = service._cluster_offsets(offsets, tolerance_ms=500)

        assert len(clusters) == 3
        assert sorted([len(c) for c in clusters]) == [2, 2, 2]

    def test_cluster_offsets_empty(self, service):
        """Test clustering with empty input."""
        clusters = service._cluster_offsets([])
        assert clusters == []

    def test_cluster_offsets_single_value(self, service):
        """Test clustering with single offset."""
        clusters = service._cluster_offsets([5000])
        assert len(clusters) == 1
        assert clusters[0] == [5000]

    def test_calculate_offset_from_matches(self, service):
        """Test calculating consensus offset from matches."""
        matches = [
            Match(offset_ms=5000, similarity=0.9),
            Match(offset_ms=5100, similarity=0.88),
            Match(offset_ms=4900, similarity=0.92),
            Match(offset_ms=5050, similarity=0.87),
            Match(offset_ms=4950, similarity=0.91),
        ]

        offset, confidence = service._calculate_offset_from_matches(matches)

        # Median should be close to 5000
        assert 4900 <= offset <= 5100
        # High confidence for 5+ matches with tight clustering
        assert confidence >= 0.85


class TestConfidenceScoring:
    """Tests for confidence scoring logic."""

    def test_confidence_for_matches_five_plus(self, service):
        """Test confidence for 5+ matches is in 0.85-0.95 range."""
        assert 0.85 <= service._confidence_for_matches(5) <= 0.95
        assert 0.85 <= service._confidence_for_matches(10) <= 0.95
        assert 0.85 <= service._confidence_for_matches(20) <= 0.95

    def test_confidence_for_matches_two_to_four(self, service):
        """Test confidence for 2-4 matches is in 0.7-0.85 range."""
        assert 0.7 <= service._confidence_for_matches(2) <= 0.85
        assert 0.7 <= service._confidence_for_matches(3) <= 0.85
        assert 0.7 <= service._confidence_for_matches(4) <= 0.85

    def test_confidence_for_matches_single(self, service):
        """Test confidence for 1 match is in 0.5-0.7 range."""
        assert 0.5 <= service._confidence_for_matches(1) <= 0.7

    def test_confidence_for_matches_zero(self, service):
        """Test confidence for 0 matches is 0."""
        assert service._confidence_for_matches(0) == 0.0


class TestAlignByTranscript:
    """Tests for transcript-based alignment."""

    def test_align_by_transcript_with_known_offset(
        self, service, sample_transcript_a, sample_transcript_b
    ):
        """Test alignment with known offset between transcripts."""
        offset, confidence, match_count = service._align_by_transcript(
            sample_transcript_a, sample_transcript_b
        )

        # B starts 5 seconds after A, so offset should be ~5000ms
        assert 4500 <= offset <= 5500
        assert match_count == 5
        assert confidence >= 0.85  # High confidence for 5+ matches

    def test_align_by_transcript_no_overlap(self, service):
        """Test alignment when transcripts have no overlapping text."""
        transcript_a = RawTranscript(
            id="transcript:source:test_a",
            source_id="source:test_a",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Alpha bravo charlie delta echo",
                    start_time=10.0,
                    end_time=13.0,
                ),
            ],
        )
        transcript_b = RawTranscript(
            id="transcript:source:test_b",
            source_id="source:test_b",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Foxtrot golf hotel india juliet",
                    start_time=10.0,
                    end_time=13.0,
                ),
            ],
        )

        offset, confidence, match_count = service._align_by_transcript(
            transcript_a, transcript_b
        )

        assert confidence == 0.0
        assert match_count == 0


class TestAlignByMetadata:
    """Tests for metadata-based alignment fallback."""

    def test_align_by_metadata_no_files(self, service):
        """Test metadata alignment when source has no files."""
        source = Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            files=[],
        )

        offset, confidence = service._align_by_metadata(source)

        assert offset == 0
        assert 0.3 <= confidence <= 0.5

    def test_align_by_metadata_confidence_range(self, service, sample_source_gopro_01):
        """Test that metadata alignment gives confidence in 0.3-0.5 range."""
        offset, confidence = service._align_by_metadata(sample_source_gopro_01)

        assert 0.3 <= confidence <= 0.5


class TestAlignSources:
    """Tests for the main align_sources method."""

    def test_align_sources_single_source(
        self, service, sample_deployment, sample_source_gopro_01
    ):
        """Test alignment with single source."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=["source:deploy:20250119_vinci_01/gopro_01"],
        )

        results = service.align_sources(
            deployment,
            sources=[sample_source_gopro_01],
            transcripts=[],
        )

        assert len(results) == 1
        assert results[sample_source_gopro_01.id].canonical_offset_ms == 0
        assert results[sample_source_gopro_01.id].confidence == 1.0
        assert results[sample_source_gopro_01.id].method == "single_source"

    def test_align_sources_multiple_with_transcripts(
        self,
        service,
        sample_deployment,
        sample_source_gopro_01,
        sample_source_gopro_02,
        sample_transcript_a,
        sample_transcript_b,
    ):
        """Test alignment with multiple sources and transcripts."""
        results = service.align_sources(
            sample_deployment,
            sources=[sample_source_gopro_01, sample_source_gopro_02],
            transcripts=[sample_transcript_a, sample_transcript_b],
        )

        assert len(results) == 2

        # Reference source (gopro_01) should have offset 0
        assert results[sample_source_gopro_01.id].canonical_offset_ms == 0
        assert results[sample_source_gopro_01.id].confidence == 1.0

        # Second source should have offset ~5000ms
        gopro02_result = results[sample_source_gopro_02.id]
        assert 4500 <= gopro02_result.canonical_offset_ms <= 5500
        assert gopro02_result.confidence >= 0.85
        assert gopro02_result.method == "transcript"
        assert gopro02_result.match_count == 5

    def test_align_sources_fallback_to_metadata(
        self, service, sample_deployment, sample_source_gopro_01, sample_source_gopro_02
    ):
        """Test that alignment falls back to metadata when no transcript overlap."""
        # Empty transcripts - should fall back to metadata
        results = service.align_sources(
            sample_deployment,
            sources=[sample_source_gopro_01, sample_source_gopro_02],
            transcripts=[],
        )

        assert len(results) == 2

        # Second source should use metadata alignment
        gopro02_result = results[sample_source_gopro_02.id]
        assert gopro02_result.method == "metadata"
        assert 0.3 <= gopro02_result.confidence <= 0.5

    def test_align_sources_empty_sources(self, service, sample_deployment):
        """Test alignment with no sources."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[],
        )

        results = service.align_sources(deployment, sources=[], transcripts=[])

        assert results == {}


class TestCalculateCanonicalTimeline:
    """Tests for calculate_canonical_timeline method."""

    def test_calculate_canonical_timeline_updates_sources(
        self, service, temp_data_dir, sample_source_gopro_01, sample_source_gopro_02
    ):
        """Test that calculate_canonical_timeline updates source files."""
        # Create deployment directory structure
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[sample_source_gopro_01.id, sample_source_gopro_02.id],
        )

        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "sources" / "gopro_01").mkdir(parents=True)
        (deploy_dir / "sources" / "gopro_02").mkdir(parents=True)

        # Write initial source files
        (deploy_dir / "sources" / "gopro_01" / "source.json").write_text(
            sample_source_gopro_01.model_dump_json()
        )
        (deploy_dir / "sources" / "gopro_02" / "source.json").write_text(
            sample_source_gopro_02.model_dump_json()
        )

        # Create alignments
        alignments = {
            sample_source_gopro_01.id: AlignmentResult(
                source_id=sample_source_gopro_01.id,
                canonical_offset_ms=0,
                confidence=1.0,
                method="transcript",
                match_count=5,
            ),
            sample_source_gopro_02.id: AlignmentResult(
                source_id=sample_source_gopro_02.id,
                canonical_offset_ms=5000,
                confidence=0.9,
                method="transcript",
                match_count=5,
            ),
        }

        # Apply alignments
        service.calculate_canonical_timeline(deployment, alignments)

        # Verify source files were updated
        gopro01_data = json.loads(
            (deploy_dir / "sources" / "gopro_01" / "source.json").read_text()
        )
        gopro02_data = json.loads(
            (deploy_dir / "sources" / "gopro_02" / "source.json").read_text()
        )

        assert gopro01_data["canonical_offset_ms"] == 0
        assert gopro01_data["alignment_confidence"] == 1.0
        assert gopro01_data["alignment_method"] == "transcript"

        assert gopro02_data["canonical_offset_ms"] == 5000
        assert gopro02_data["alignment_confidence"] == 0.9
        assert gopro02_data["alignment_method"] == "transcript"


class TestVerifyAlignment:
    """Tests for alignment verification."""

    def test_verify_alignment_good_alignment(
        self, service, sample_transcript_a, sample_transcript_b
    ):
        """Test verification with well-aligned transcripts."""
        # These alignments should make the transcripts line up correctly
        alignments = {
            sample_transcript_a.source_id: AlignmentResult(
                source_id=sample_transcript_a.source_id,
                canonical_offset_ms=0,
                confidence=1.0,
                method="transcript",
            ),
            sample_transcript_b.source_id: AlignmentResult(
                source_id=sample_transcript_b.source_id,
                canonical_offset_ms=5000,  # Correct offset
                confidence=0.9,
                method="transcript",
            ),
        }

        issues = service.verify_alignment(
            alignments,
            [sample_transcript_a, sample_transcript_b],
            tolerance_ms=1000,
        )

        # Should have no major issues since alignment is correct
        assert len([i for i in issues if i.severity == "error"]) == 0

    def test_verify_alignment_bad_alignment(
        self, service, sample_transcript_a, sample_transcript_b
    ):
        """Test verification with misaligned transcripts."""
        # Intentionally wrong offset
        alignments = {
            sample_transcript_a.source_id: AlignmentResult(
                source_id=sample_transcript_a.source_id,
                canonical_offset_ms=0,
                confidence=1.0,
                method="transcript",
            ),
            sample_transcript_b.source_id: AlignmentResult(
                source_id=sample_transcript_b.source_id,
                canonical_offset_ms=0,  # Should be 5000, this is wrong
                confidence=0.5,
                method="metadata",
            ),
        }

        issues = service.verify_alignment(
            alignments,
            [sample_transcript_a, sample_transcript_b],
            tolerance_ms=1000,
        )

        # Should find issues due to misalignment
        assert len(issues) > 0


class TestSaveLoadAlignmentResults:
    """Tests for saving and loading alignment results."""

    def test_save_and_load_alignment_results(
        self, service, temp_data_dir, sample_source_gopro_01
    ):
        """Test saving and loading alignment results."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[sample_source_gopro_01.id],
        )

        # Create deployment directory structure
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "cache" / "alignment").mkdir(parents=True)

        alignments = {
            sample_source_gopro_01.id: AlignmentResult(
                source_id=sample_source_gopro_01.id,
                canonical_offset_ms=0,
                confidence=1.0,
                method="single_source",
                match_count=0,
            ),
        }

        # Save
        service.save_alignment_results(deployment, alignments)

        # Load
        loaded = service.load_alignment_results(deployment)

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[sample_source_gopro_01.id].canonical_offset_ms == 0
        assert loaded[sample_source_gopro_01.id].confidence == 1.0

    def test_load_nonexistent_alignment_results(self, service, temp_data_dir):
        """Test loading when no cached results exist."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[],
        )

        result = service.load_alignment_results(deployment)
        assert result is None


class TestLCSLength:
    """Tests for longest common subsequence helper."""

    def test_lcs_identical_strings(self, service):
        """Test LCS for identical strings."""
        result = service._lcs_length("hello", "hello")
        assert result == 5

    def test_lcs_no_common(self, service):
        """Test LCS with no common characters."""
        result = service._lcs_length("abc", "xyz")
        assert result == 0

    def test_lcs_partial_overlap(self, service):
        """Test LCS with partial overlap."""
        result = service._lcs_length("abcdef", "acdf")
        assert result == 4  # 'a', 'c', 'd', 'f'

    def test_lcs_empty_string(self, service):
        """Test LCS with empty string."""
        assert service._lcs_length("", "test") == 0
        assert service._lcs_length("test", "") == 0
        assert service._lcs_length("", "") == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_align_sources_with_short_segments(self, service):
        """Test alignment with very short transcript segments."""
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[
                "source:deploy:20250119_vinci_01/gopro_01",
                "source:deploy:20250119_vinci_01/gopro_02",
            ],
        )

        source_a = Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            files=[
                SourceFile(
                    filename="test.mp4",
                    file_path="/path/test.mp4",
                    duration_seconds=60.0,
                    start_offset_ms=0,
                )
            ],
        )

        source_b = Source(
            id="source:deploy:20250119_vinci_01/gopro_02",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=2,
            files=[
                SourceFile(
                    filename="test2.mp4",
                    file_path="/path/test2.mp4",
                    duration_seconds=60.0,
                    start_offset_ms=0,
                )
            ],
        )

        # Transcripts with only short segments (< 3 words) - should be filtered out
        transcript_a = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(text="Yes", start_time=10.0, end_time=11.0),
                TranscriptSegment(text="Okay", start_time=20.0, end_time=21.0),
            ],
        )

        transcript_b = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_02",
            source_id="source:deploy:20250119_vinci_01/gopro_02",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(text="Yes", start_time=10.0, end_time=11.0),
                TranscriptSegment(text="Okay", start_time=20.0, end_time=21.0),
            ],
        )

        results = service.align_sources(
            deployment,
            sources=[source_a, source_b],
            transcripts=[transcript_a, transcript_b],
        )

        # Should fall back to metadata due to no valid matches
        assert results[source_b.id].method == "metadata"

    def test_align_conflicting_matches(self, service):
        """Test alignment with conflicting offset matches."""
        transcript_a = RawTranscript(
            id="transcript:source:test_a",
            source_id="source:test_a",
            transcription_service="elevenlabs",
            segments=[
                # Multiple matches with different offsets
                TranscriptSegment(
                    text="This is the first test phrase here",
                    start_time=10.0,
                    end_time=13.0,
                ),
                TranscriptSegment(
                    text="This is the second test phrase here",
                    start_time=20.0,
                    end_time=23.0,
                ),
                TranscriptSegment(
                    text="This is the third test phrase here",
                    start_time=30.0,
                    end_time=33.0,
                ),
            ],
        )

        transcript_b = RawTranscript(
            id="transcript:source:test_b",
            source_id="source:test_b",
            transcription_service="elevenlabs",
            segments=[
                # Same phrases but at wildly different offsets
                TranscriptSegment(
                    text="This is the first test phrase here",
                    start_time=5.0,  # offset = 5000
                    end_time=8.0,
                ),
                TranscriptSegment(
                    text="This is the second test phrase here",
                    start_time=200.0,  # offset = -180000 (conflicting!)
                    end_time=203.0,
                ),
                TranscriptSegment(
                    text="This is the third test phrase here",
                    start_time=25.0,  # offset = 5000
                    end_time=28.0,
                ),
            ],
        )

        offset, confidence, match_count = service._align_by_transcript(
            transcript_a, transcript_b
        )

        # Should use clustering to find the dominant offset (~5000)
        # and ignore the outlier
        assert match_count == 3
        # Confidence should be reduced due to conflicting matches
        assert confidence < 0.95
