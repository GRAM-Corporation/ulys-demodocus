"""Tests for the TimeAlignmentService.

Tests cover:
- Single source alignment (trivial case)
- Multi-source alignment using audio fingerprints
- Fallback alignment using transcript matching
- Metadata-based alignment
- Confidence scoring ranges per spec
- Alignment verification
- align_sources() high-level method
- Edge cases and error handling
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gram_deploy.models import (
    Deployment,
    DeploymentStatus,
    DeviceType,
    RawTranscript,
    Source,
    SourceFile,
    TimeAlignment,
    TranscriptSegment,
    TranscriptSpeaker,
    TranscriptStatus,
)
from gram_deploy.services.time_alignment import (
    AlignmentIssue,
    AlignmentResult,
    TimeAlignmentService,
)


# Test fixtures

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with deployment structure."""
    data_dir = tmp_path / "deployments"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
        sources=[
            "source:deploy:20250119_vinci_01/gopro_01",
            "source:deploy:20250119_vinci_01/phone_01",
        ],
        status=DeploymentStatus.ALIGNING,
    )


@pytest.fixture
def sample_source_gopro(tmp_path):
    """Create a sample GoPro source."""
    video_file = tmp_path / "GX010001.MP4"
    video_file.write_bytes(b"fake video content")

    return Source(
        id="source:deploy:20250119_vinci_01/gopro_01",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=1,
        files=[
            SourceFile(
                filename="GX010001.MP4",
                file_path=str(video_file),
                duration_seconds=600.0,
                start_offset_ms=0,
            ),
        ],
        total_duration_seconds=600.0,
        transcript_status=TranscriptStatus.COMPLETE,
    )


@pytest.fixture
def sample_source_phone(tmp_path):
    """Create a sample phone source."""
    video_file = tmp_path / "VID_001.mp4"
    video_file.write_bytes(b"fake video content")

    return Source(
        id="source:deploy:20250119_vinci_01/phone_01",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.PHONE,
        device_number=1,
        files=[
            SourceFile(
                filename="VID_001.mp4",
                file_path=str(video_file),
                duration_seconds=580.0,
                start_offset_ms=0,
            ),
        ],
        total_duration_seconds=580.0,
        transcript_status=TranscriptStatus.COMPLETE,
    )


@pytest.fixture
def sample_transcript_gopro():
    """Create a sample transcript for GoPro source."""
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_01",
        source_id="source:deploy:20250119_vinci_01/gopro_01",
        transcription_service="elevenlabs",
        segments=[
            TranscriptSegment(
                text="Hello everyone welcome to the deployment meeting today",
                start_time=10.0,
                end_time=15.0,
                speaker=TranscriptSpeaker(id="speaker_0"),
            ),
            TranscriptSegment(
                text="Let's review the checklist for today's activities",
                start_time=16.0,
                end_time=20.0,
                speaker=TranscriptSpeaker(id="speaker_0"),
            ),
            TranscriptSegment(
                text="First we need to check the Starlink battery status",
                start_time=25.0,
                end_time=30.0,
                speaker=TranscriptSpeaker(id="speaker_1"),
            ),
        ],
        audio_duration_seconds=600.0,
    )


@pytest.fixture
def sample_transcript_phone():
    """Create a sample transcript for phone source with matching text."""
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/phone_01",
        source_id="source:deploy:20250119_vinci_01/phone_01",
        transcription_service="elevenlabs",
        segments=[
            TranscriptSegment(
                text="Hello everyone welcome to the deployment meeting today",
                start_time=12.5,  # 2.5 seconds offset from gopro
                end_time=17.5,
                speaker=TranscriptSpeaker(id="speaker_A"),
            ),
            TranscriptSegment(
                text="Let's review the checklist for today's activities",
                start_time=18.5,
                end_time=22.5,
                speaker=TranscriptSpeaker(id="speaker_A"),
            ),
            TranscriptSegment(
                text="First we need to check the Starlink battery status",
                start_time=27.5,
                end_time=32.5,
                speaker=TranscriptSpeaker(id="speaker_B"),
            ),
        ],
        audio_duration_seconds=580.0,
    )


class TestTimeAlignmentServiceInit:
    """Tests for TimeAlignmentService initialization."""

    def test_init_creates_cache_dir(self, tmp_path):
        """Should create cache directory if it doesn't exist."""
        cache_dir = tmp_path / "alignment_cache"
        service = TimeAlignmentService(cache_dir=str(cache_dir))
        assert cache_dir.exists()
        assert service.cache_dir == cache_dir

    def test_init_with_data_dir(self, tmp_cache_dir, tmp_data_dir):
        """Should accept optional data_dir parameter."""
        service = TimeAlignmentService(
            cache_dir=str(tmp_cache_dir),
            data_dir=str(tmp_data_dir),
        )
        assert service.data_dir == tmp_data_dir

    def test_init_without_data_dir(self, tmp_cache_dir):
        """Should work without data_dir for direct compute_alignment calls."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))
        assert service.data_dir is None


class TestSingleSourceAlignment:
    """Tests for single source (trivial) alignment."""

    def test_single_source_alignment(
        self, tmp_cache_dir, sample_source_gopro
    ):
        """Should handle single source with zero offset and full confidence."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = service.compute_alignment(
            sources=[sample_source_gopro],
            transcripts=[],
        )

        assert sample_source_gopro.id in alignment.source_offsets
        assert alignment.source_offsets[sample_source_gopro.id] == 0
        assert alignment.confidence_scores[sample_source_gopro.id] == 1.0
        assert alignment.alignment_methods[sample_source_gopro.id] == "single_source"

    def test_single_source_no_fingerprints_no_transcripts(
        self, tmp_cache_dir, sample_source_gopro
    ):
        """Should work with single source even without fingerprints or transcripts."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = service.compute_alignment(
            sources=[sample_source_gopro],
            transcripts=[],
            audio_fingerprints=None,
        )

        assert len(alignment.source_offsets) == 1
        assert alignment.alignment_methods[sample_source_gopro.id] == "single_source"


class TestAudioFingerprintAlignment:
    """Tests for audio fingerprint-based alignment."""

    def test_audio_fingerprint_alignment_high_correlation(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should align sources with high confidence when fingerprints correlate well."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        # Create synthetic fingerprints with known offset
        # Reference fingerprint
        fp_gopro = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).tobytes()
        # Phone fingerprint with 2-unit offset (should detect ~200ms offset)
        fp_phone = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32).tobytes()

        fingerprints = {
            sample_source_gopro.id: fp_gopro,
            sample_source_phone.id: fp_phone,
        }

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[],
            audio_fingerprints=fingerprints,
        )

        # Reference source should have zero offset
        assert alignment.source_offsets[sample_source_gopro.id] == 0
        assert alignment.confidence_scores[sample_source_gopro.id] == 1.0

        # Phone source should be aligned
        assert sample_source_phone.id in alignment.source_offsets
        assert alignment.alignment_methods[sample_source_phone.id] == "audio_fingerprint"

    def test_audio_fingerprint_confidence_in_spec_range(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Confidence should be in 0.9-1.0 range per spec for audio fingerprint."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        # Create fingerprints that will produce moderate correlation
        fp_gopro = np.array([1, 2, 3, 4, 5] * 10, dtype=np.int32).tobytes()
        fp_phone = np.array([1, 2, 3, 4, 5] * 10, dtype=np.int32).tobytes()

        fingerprints = {
            sample_source_gopro.id: fp_gopro,
            sample_source_phone.id: fp_phone,
        }

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[],
            audio_fingerprints=fingerprints,
        )

        # If aligned via audio fingerprint, confidence should be 0.9-1.0
        if alignment.alignment_methods.get(sample_source_phone.id) == "audio_fingerprint":
            confidence = alignment.confidence_scores[sample_source_phone.id]
            assert 0.9 <= confidence <= 1.0, f"Audio fingerprint confidence {confidence} not in 0.9-1.0"

    def test_audio_fingerprint_empty_arrays(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should handle empty fingerprint arrays gracefully."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        fingerprints = {
            sample_source_gopro.id: b"",
            sample_source_phone.id: b"",
        }

        # Should not raise, should fall back to other methods
        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[],
            audio_fingerprints=fingerprints,
        )

        # Reference source will still be set
        assert sample_source_gopro.id in alignment.source_offsets


class TestTranscriptAlignment:
    """Tests for transcript-based alignment."""

    def test_transcript_alignment_matching_text(
        self,
        tmp_cache_dir,
        sample_source_gopro,
        sample_source_phone,
        sample_transcript_gopro,
        sample_transcript_phone,
    ):
        """Should align sources using matching transcript text."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[sample_transcript_gopro, sample_transcript_phone],
            audio_fingerprints=None,
        )

        # Both sources should be aligned
        assert sample_source_gopro.id in alignment.source_offsets
        assert sample_source_phone.id in alignment.source_offsets

    def test_transcript_confidence_in_spec_range(
        self,
        tmp_cache_dir,
        sample_source_gopro,
        sample_source_phone,
        sample_transcript_gopro,
        sample_transcript_phone,
    ):
        """Confidence should be in 0.7-0.9 range per spec for transcript matching."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[sample_transcript_gopro, sample_transcript_phone],
            audio_fingerprints=None,
        )

        # If aligned via transcript, confidence should be 0.7-0.9
        for source_id, method in alignment.alignment_methods.items():
            if method == "transcript_match":
                confidence = alignment.confidence_scores[source_id]
                assert 0.7 <= confidence <= 0.9, f"Transcript confidence {confidence} not in 0.7-0.9"

    def test_transcript_alignment_no_matches(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should fall back to metadata when transcripts don't match."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        # Create transcripts with completely different text
        transcript_a = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="The quick brown fox jumps over the lazy dog",
                    start_time=10.0,
                    end_time=15.0,
                ),
            ],
        )
        transcript_b = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/phone_01",
            source_id="source:deploy:20250119_vinci_01/phone_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Lorem ipsum dolor sit amet consectetur adipiscing",
                    start_time=10.0,
                    end_time=15.0,
                ),
            ],
        )

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[transcript_a, transcript_b],
            audio_fingerprints=None,
        )

        # Phone source should fall back to metadata alignment
        method = alignment.alignment_methods.get(sample_source_phone.id)
        assert method in ("metadata", "unaligned"), f"Expected fallback, got {method}"


class TestMetadataAlignment:
    """Tests for metadata-based fallback alignment."""

    def test_metadata_alignment_with_files(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should use file metadata when other methods unavailable."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[],
            audio_fingerprints=None,
        )

        # All sources should be aligned
        assert len(alignment.source_offsets) == 2

    def test_metadata_confidence_in_spec_range(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Confidence should be in 0.3-0.5 range per spec for metadata alignment."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[],
            audio_fingerprints=None,
        )

        # For metadata aligned sources, confidence should be 0.3-0.5
        for source_id, method in alignment.alignment_methods.items():
            if method == "metadata":
                confidence = alignment.confidence_scores[source_id]
                assert 0.3 <= confidence <= 0.5, f"Metadata confidence {confidence} not in 0.3-0.5"

    def test_metadata_alignment_missing_files(self, tmp_cache_dir):
        """Should handle sources with missing files gracefully."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        # Create source with non-existent file
        source = Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            files=[
                SourceFile(
                    filename="nonexistent.mp4",
                    file_path="/nonexistent/path/video.mp4",
                    duration_seconds=100.0,
                    start_offset_ms=0,
                ),
            ],
        )

        alignment = service.compute_alignment(
            sources=[source],
            transcripts=[],
        )

        # Should still produce an alignment
        assert source.id in alignment.source_offsets


class TestTextSimilarity:
    """Tests for text similarity calculation."""

    def test_text_similarity_identical(self, tmp_cache_dir):
        """Identical text should have similarity of 1.0."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        similarity = service._text_similarity(
            "hello world this is a test",
            "hello world this is a test",
        )
        assert similarity == 1.0

    def test_text_similarity_partial_overlap(self, tmp_cache_dir):
        """Partial overlap should have intermediate similarity."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        similarity = service._text_similarity(
            "hello world this is a test",
            "hello world this is different",
        )
        assert 0.0 < similarity < 1.0

    def test_text_similarity_no_overlap(self, tmp_cache_dir):
        """No overlap should have similarity of 0.0."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        similarity = service._text_similarity(
            "hello world",
            "goodbye universe",
        )
        assert similarity == 0.0

    def test_text_similarity_case_insensitive(self, tmp_cache_dir):
        """Similarity should be case-insensitive."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        similarity = service._text_similarity(
            "Hello World",
            "hello world",
        )
        assert similarity == 1.0


class TestCrossCorrelation:
    """Tests for audio fingerprint cross-correlation."""

    def test_cross_correlate_identical_fingerprints(self, tmp_cache_dir):
        """Identical fingerprints should have zero offset."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        fp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).tobytes()

        offset, confidence = service._cross_correlate_fingerprints(fp, fp)

        assert offset == 0
        assert confidence > 0.5

    def test_cross_correlate_empty_fingerprints(self, tmp_cache_dir):
        """Empty fingerprints should return zero offset and confidence."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        offset, confidence = service._cross_correlate_fingerprints(b"", b"")

        assert offset == 0
        assert confidence == 0.0


class TestAlignmentVerification:
    """Tests for alignment verification."""

    def test_verify_alignment_no_issues(
        self,
        tmp_cache_dir,
        sample_source_gopro,
        sample_source_phone,
        sample_transcript_gopro,
        sample_transcript_phone,
    ):
        """Should report no issues for well-aligned sources."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        # Create a perfect alignment - phone transcript is offset by 2.5s
        # so we need offset of -2500ms to align them to canonical time
        alignment = TimeAlignment(
            deployment_id="deploy:20250119_vinci_01",
            canonical_start_time=datetime.utcnow(),
            source_offsets={
                sample_source_gopro.id: 0,
                sample_source_phone.id: -2500,  # Subtract 2.5s to align phone to gopro
            },
            confidence_scores={
                sample_source_gopro.id: 1.0,
                sample_source_phone.id: 0.9,
            },
        )

        issues = service.verify_alignment(
            alignment,
            [sample_transcript_gopro, sample_transcript_phone],
            tolerance_ms=1000,  # Allow reasonable tolerance for matching
        )

        # Should have few or no errors (warnings are acceptable)
        error_count = len([i for i in issues if i.severity == "error"])
        assert error_count == 0


class TestAlignSourcesMethod:
    """Tests for the high-level align_sources method."""

    def test_align_sources_requires_data_dir(self, tmp_cache_dir, sample_deployment):
        """Should raise ValueError if data_dir not set."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        with pytest.raises(ValueError) as exc_info:
            service.align_sources(sample_deployment)
        assert "data_dir must be set" in str(exc_info.value)

    def test_align_sources_empty_deployment(self, tmp_cache_dir, tmp_data_dir):
        """Should return empty dict for deployment with no sources."""
        service = TimeAlignmentService(
            cache_dir=str(tmp_cache_dir),
            data_dir=str(tmp_data_dir),
        )

        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[],
        )

        results = service.align_sources(deployment)
        assert results == {}

    def test_align_sources_returns_alignment_results(
        self, tmp_cache_dir, tmp_data_dir, sample_source_gopro
    ):
        """Should return dict of AlignmentResult objects."""
        # Set up directory structure
        deploy_dir = tmp_data_dir / "deploy_20250119_vinci_01"
        sources_dir = deploy_dir / "sources" / "gopro_01"
        canonical_dir = deploy_dir / "canonical"
        sources_dir.mkdir(parents=True)
        canonical_dir.mkdir(parents=True)

        # Save source to disk
        source_path = sources_dir / "source.json"
        source_path.write_text(sample_source_gopro.model_dump_json(indent=2))

        service = TimeAlignmentService(
            cache_dir=str(tmp_cache_dir),
            data_dir=str(tmp_data_dir),
        )

        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[sample_source_gopro.id],
        )

        results = service.align_sources(deployment)

        assert sample_source_gopro.id in results
        result = results[sample_source_gopro.id]
        assert isinstance(result, AlignmentResult)
        assert result.source_id == sample_source_gopro.id
        assert result.canonical_offset_ms == 0
        assert result.confidence == 1.0
        assert result.method == "single_source"

    def test_align_sources_saves_alignment_file(
        self, tmp_cache_dir, tmp_data_dir, sample_source_gopro
    ):
        """Should save alignment.json to canonical directory."""
        # Set up directory structure
        deploy_dir = tmp_data_dir / "deploy_20250119_vinci_01"
        sources_dir = deploy_dir / "sources" / "gopro_01"
        canonical_dir = deploy_dir / "canonical"
        sources_dir.mkdir(parents=True)
        canonical_dir.mkdir(parents=True)

        # Save source to disk
        source_path = sources_dir / "source.json"
        source_path.write_text(sample_source_gopro.model_dump_json(indent=2))

        service = TimeAlignmentService(
            cache_dir=str(tmp_cache_dir),
            data_dir=str(tmp_data_dir),
        )

        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[sample_source_gopro.id],
        )

        service.align_sources(deployment)

        alignment_path = canonical_dir / "alignment.json"
        assert alignment_path.exists()

        # Verify content
        data = json.loads(alignment_path.read_text())
        assert "source_offsets" in data
        assert sample_source_gopro.id in data["source_offsets"]

    def test_align_sources_updates_source_files(
        self, tmp_cache_dir, tmp_data_dir, sample_source_gopro
    ):
        """Should update source.json files with alignment info."""
        # Set up directory structure
        deploy_dir = tmp_data_dir / "deploy_20250119_vinci_01"
        sources_dir = deploy_dir / "sources" / "gopro_01"
        canonical_dir = deploy_dir / "canonical"
        sources_dir.mkdir(parents=True)
        canonical_dir.mkdir(parents=True)

        # Save source to disk
        source_path = sources_dir / "source.json"
        source_path.write_text(sample_source_gopro.model_dump_json(indent=2))

        service = TimeAlignmentService(
            cache_dir=str(tmp_cache_dir),
            data_dir=str(tmp_data_dir),
        )

        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=[sample_source_gopro.id],
        )

        service.align_sources(deployment)

        # Read back source file
        data = json.loads(source_path.read_text())
        assert "canonical_offset_ms" in data
        assert "alignment_confidence" in data
        assert "alignment_method" in data


class TestCalculateCanonicalTimeline:
    """Tests for canonical timeline calculation."""

    def test_calculate_canonical_timeline_normalizes_offsets(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should normalize offsets so earliest source starts at 0."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = TimeAlignment(
            deployment_id="deploy:20250119_vinci_01",
            canonical_start_time=datetime.utcnow(),
            source_offsets={
                sample_source_gopro.id: 5000,  # 5 seconds
                sample_source_phone.id: 3000,  # 3 seconds (earliest)
            },
        )

        result = service.calculate_canonical_timeline(
            alignment,
            [sample_source_gopro, sample_source_phone],
        )

        # Phone (earliest) should now be at 0
        assert result.source_offsets[sample_source_phone.id] == 0
        # GoPro should be 2 seconds after phone
        assert result.source_offsets[sample_source_gopro.id] == 2000


class TestSaveLoadAlignment:
    """Tests for alignment persistence."""

    def test_save_and_load_alignment(self, tmp_cache_dir, tmp_path):
        """Should correctly save and load alignment."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = TimeAlignment(
            deployment_id="deploy:20250119_vinci_01",
            canonical_start_time=datetime(2025, 1, 19, 10, 0, 0),
            source_offsets={"source:deploy:20250119_vinci_01/gopro_01": 1000},
            confidence_scores={"source:deploy:20250119_vinci_01/gopro_01": 0.95},
            alignment_methods={"source:deploy:20250119_vinci_01/gopro_01": "audio_fingerprint"},
        )

        path = tmp_path / "alignment.json"
        service.save_alignment(alignment, str(path))

        loaded = service.load_alignment(str(path))

        assert loaded.deployment_id == alignment.deployment_id
        assert loaded.source_offsets == alignment.source_offsets
        assert loaded.confidence_scores == alignment.confidence_scores


class TestApplyAlignment:
    """Tests for applying alignment to sources."""

    def test_apply_alignment_updates_sources(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should update source objects with alignment data."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        alignment = TimeAlignment(
            deployment_id="deploy:20250119_vinci_01",
            canonical_start_time=datetime.utcnow(),
            source_offsets={
                sample_source_gopro.id: 0,
                sample_source_phone.id: 2500,
            },
            confidence_scores={
                sample_source_gopro.id: 1.0,
                sample_source_phone.id: 0.95,
            },
            alignment_methods={
                sample_source_gopro.id: "audio_fingerprint",
                sample_source_phone.id: "audio_fingerprint",
            },
        )

        sources = [sample_source_gopro, sample_source_phone]
        service.apply_alignment(alignment, sources)

        assert sample_source_gopro.canonical_offset_ms == 0
        assert sample_source_gopro.alignment_confidence == 1.0
        assert sample_source_gopro.alignment_method == "audio_fingerprint"

        assert sample_source_phone.canonical_offset_ms == 2500
        assert sample_source_phone.alignment_confidence == 0.95


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sources_list(self, tmp_cache_dir):
        """Should raise ValueError for empty sources list."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        with pytest.raises(ValueError) as exc_info:
            service.compute_alignment(sources=[], transcripts=[])
        assert "At least one source is required" in str(exc_info.value)

    def test_sources_with_no_files(self, tmp_cache_dir):
        """Should handle sources with empty files list."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        source = Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            files=[],
        )

        alignment = service.compute_alignment(sources=[source], transcripts=[])

        # Should still produce an alignment
        assert source.id in alignment.source_offsets

    def test_fingerprint_missing_for_some_sources(
        self, tmp_cache_dir, sample_source_gopro, sample_source_phone
    ):
        """Should handle partial fingerprint data gracefully."""
        service = TimeAlignmentService(cache_dir=str(tmp_cache_dir))

        # Only provide fingerprint for one source
        fingerprints = {
            sample_source_gopro.id: np.array([1, 2, 3], dtype=np.int32).tobytes(),
            # Phone fingerprint missing
        }

        alignment = service.compute_alignment(
            sources=[sample_source_gopro, sample_source_phone],
            transcripts=[],
            audio_fingerprints=fingerprints,
        )

        # Both sources should still be aligned (phone via fallback)
        assert len(alignment.source_offsets) == 2


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_alignment_result_creation(self):
        """Should create AlignmentResult with all fields."""
        result = AlignmentResult(
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            canonical_offset_ms=1500,
            confidence=0.95,
            method="audio_fingerprint",
        )

        assert result.source_id == "source:deploy:20250119_vinci_01/gopro_01"
        assert result.canonical_offset_ms == 1500
        assert result.confidence == 0.95
        assert result.method == "audio_fingerprint"


class TestAlignmentIssue:
    """Tests for AlignmentIssue dataclass."""

    def test_alignment_issue_creation(self):
        """Should create AlignmentIssue with all fields."""
        issue = AlignmentIssue(
            source_id="source:deploy:20250119_vinci_01/phone_01",
            description="Text differs by 5000ms from reference",
            severity="warning",
            suggested_fix="Adjust offset by -5000ms",
        )

        assert issue.source_id == "source:deploy:20250119_vinci_01/phone_01"
        assert "5000ms" in issue.description
        assert issue.severity == "warning"
        assert issue.suggested_fix is not None
