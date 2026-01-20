"""Tests for the TranscriptionService.

Tests cover:
- S3 presigned URL generation
- Provider API response parsing
- RawTranscript creation
- Error handling (rate limits, API failures, missing configuration)
- Status updates and transcript saving
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.source import Source, SourceFile, DeviceType, TranscriptStatus
from gram_deploy.models.transcript import RawTranscript, TranscriptSegment, TranscriptSpeaker, WordTiming

# Import the transcription service module directly to avoid loading all services
import importlib.util

spec = importlib.util.spec_from_file_location(
    "transcription_service",
    Path(__file__).parent.parent / "src" / "gram_deploy" / "services" / "transcription_service.py",
)
ts_module = importlib.util.module_from_spec(spec)

# We need to patch get_settings before loading the module
# Create a mock settings that will be used during module load
mock_default_settings = MagicMock()
mock_default_settings.transcription_provider.value = "elevenlabs"
mock_default_settings.get_transcription_api_key.return_value = "test-api-key"
mock_default_settings.data_dir = Path("/tmp/test-data")
mock_default_settings.s3_bucket = "test-bucket"
mock_default_settings.s3_region = "us-east-1"

# Execute the module
spec.loader.exec_module(ts_module)
TranscriptionService = ts_module.TranscriptionService
TranscriptionError = ts_module.TranscriptionError


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_mock_settings(data_dir=None, s3_bucket="test-bucket"):
    """Create a mock settings object."""
    settings = MagicMock()
    settings.transcription_provider.value = "elevenlabs"
    settings.get_transcription_api_key.return_value = "test-api-key"
    settings.data_dir = Path(data_dir) if data_dir else Path("/tmp/test-data")
    settings.s3_bucket = s3_bucket
    settings.s3_region = "us-east-1"
    return settings


@pytest.fixture
def sample_source():
    """Create a sample source for testing."""
    return Source(
        id="source:deploy:20250119_vinci_01/gopro_01",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=1,
        files=[
            SourceFile(
                filename="GX010001.MP4",
                file_path="/path/to/GX010001.MP4",
                duration_seconds=300.0,
                start_offset_ms=0,
            ),
        ],
    )


@pytest.fixture
def sample_source_multi_file():
    """Create a sample source with multiple files."""
    return Source(
        id="source:deploy:20250119_vinci_01/gopro_01",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=1,
        files=[
            SourceFile(
                filename="GX010001.MP4",
                file_path="/path/to/GX010001.MP4",
                duration_seconds=300.0,
                start_offset_ms=0,
            ),
            SourceFile(
                filename="GX010002.MP4",
                file_path="/path/to/GX010002.MP4",
                duration_seconds=300.0,
                start_offset_ms=300000,
            ),
        ],
    )


@pytest.fixture
def elevenlabs_response():
    """Sample ElevenLabs API response."""
    return {
        "language_code": "en",
        "audio_duration": 300.5,
        "words": [
            {"text": "Hello", "start": 0.0, "end": 0.5, "speaker_id": "speaker_1", "confidence": 0.95},
            {"text": "world", "start": 0.5, "end": 1.0, "speaker_id": "speaker_1", "confidence": 0.92},
            {"text": "How", "start": 2.0, "end": 2.3, "speaker_id": "speaker_2", "confidence": 0.88},
            {"text": "are", "start": 2.3, "end": 2.5, "speaker_id": "speaker_2", "confidence": 0.90},
            {"text": "you", "start": 2.5, "end": 2.8, "speaker_id": "speaker_2", "confidence": 0.91},
        ],
    }


@pytest.fixture
def assemblyai_response():
    """Sample AssemblyAI API response."""
    return {
        "id": "test-transcript-id",
        "status": "completed",
        "language_code": "en",
        "audio_duration": 300.5,
        "utterances": [
            {
                "speaker": "A",
                "text": "Hello world",
                "start": 0,
                "end": 1000,
                "confidence": 0.93,
                "words": [
                    {"text": "Hello", "start": 0, "end": 500, "confidence": 0.95},
                    {"text": "world", "start": 500, "end": 1000, "confidence": 0.92},
                ],
            },
            {
                "speaker": "B",
                "text": "How are you",
                "start": 2000,
                "end": 2800,
                "confidence": 0.89,
                "words": [
                    {"text": "How", "start": 2000, "end": 2300, "confidence": 0.88},
                    {"text": "are", "start": 2300, "end": 2500, "confidence": 0.90},
                    {"text": "you", "start": 2500, "end": 2800, "confidence": 0.91},
                ],
            },
        ],
    }


@pytest.fixture
def deepgram_response():
    """Sample Deepgram API response."""
    return {
        "metadata": {
            "language": "en",
            "duration": 300.5,
            "model_info": {"name": "nova-2"},
        },
        "results": {
            "utterances": [
                {"speaker": 0, "transcript": "Hello world", "start": 0.0, "end": 1.0, "confidence": 0.93},
                {"speaker": 1, "transcript": "How are you", "start": 2.0, "end": 2.8, "confidence": 0.89},
            ],
        },
    }


class TestTranscriptionServiceInit:
    """Tests for TranscriptionService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            assert service.provider == "elevenlabs"
            assert service.api_key == "test-api-key"
            assert service.s3_bucket == "test-bucket"
            assert service.s3_region == "us-east-1"

    def test_init_with_custom_provider(self):
        """Test initialization with custom provider."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService(provider="assemblyai", api_key="custom-key")

            assert service.provider == "assemblyai"
            assert service.api_key == "custom-key"

    def test_init_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="Unknown provider"):
                TranscriptionService(provider="invalid_provider")

    def test_init_with_custom_data_dir(self, temp_data_dir):
        """Test initialization with custom data directory."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService(data_dir=temp_data_dir)

            assert service.data_dir == Path(temp_data_dir)


class TestS3KeyGeneration:
    """Tests for _get_s3_key method."""

    def test_get_s3_key_basic(self, sample_source):
        """Test basic S3 key generation."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()
            s3_key = service._get_s3_key(sample_source, "GX010001.MP4")

            assert s3_key == "deployments/deploy_20250119_vinci_01/sources/gopro_01/GX010001.MP4"

    def test_get_s3_key_different_source(self):
        """Test S3 key generation for different source types."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            source = Source(
                id="source:deploy:20250120_rome_02/phone_01",
                deployment_id="deploy:20250120_rome_02",
                device_type=DeviceType.PHONE,
                device_number=1,
                files=[
                    SourceFile(
                        filename="video.mp4",
                        file_path="/path/to/video.mp4",
                        duration_seconds=120.0,
                        start_offset_ms=0,
                    ),
                ],
            )
            service = TranscriptionService()
            s3_key = service._get_s3_key(source, "video.mp4")

            assert s3_key == "deployments/deploy_20250120_rome_02/sources/phone_01/video.mp4"


class TestPresignedUrlGeneration:
    """Tests for _generate_presigned_url method."""

    def test_generate_presigned_url(self):
        """Test presigned URL generation with mocked S3 client."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://test-bucket.s3.amazonaws.com/test-key?signature=xxx"

            service = TranscriptionService()
            service._s3_client = mock_s3

            url = service._generate_presigned_url("test-key")

            mock_s3.generate_presigned_url.assert_called_once_with(
                "get_object",
                Params={"Bucket": "test-bucket", "Key": "test-key"},
                ExpiresIn=3600,
            )
            assert url == "https://test-bucket.s3.amazonaws.com/test-key?signature=xxx"

    def test_generate_presigned_url_custom_expiry(self):
        """Test presigned URL generation with custom expiry."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://test-url"

            service = TranscriptionService()
            service._s3_client = mock_s3

            service._generate_presigned_url("test-key", expires_in=7200)

            mock_s3.generate_presigned_url.assert_called_once_with(
                "get_object",
                Params={"Bucket": "test-bucket", "Key": "test-key"},
                ExpiresIn=7200,
            )


class TestElevenLabsTranscription:
    """Tests for ElevenLabs transcription."""

    def test_parse_elevenlabs_response(self, elevenlabs_response):
        """Test parsing ElevenLabs API response."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()
            transcript = service._parse_elevenlabs_response(elevenlabs_response)

            assert transcript.transcription_service == "elevenlabs"
            assert transcript.transcription_model == "scribe_v1"
            assert transcript.language_code == "en"
            assert transcript.audio_duration_seconds == 300.5

            # Should be merged into 2 segments (speaker change)
            assert len(transcript.segments) == 2
            assert transcript.segments[0].speaker.id == "speaker_1"
            assert transcript.segments[1].speaker.id == "speaker_2"

    def test_transcribe_elevenlabs_success(self, elevenlabs_response):
        """Test successful ElevenLabs transcription."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = elevenlabs_response
            service._client.post = MagicMock(return_value=mock_response)

            transcript = service._transcribe_elevenlabs("https://test-url")

            assert transcript.transcription_service == "elevenlabs"
            service._client.post.assert_called_once()

    def test_transcribe_elevenlabs_rate_limit(self, elevenlabs_response):
        """Test ElevenLabs rate limit handling."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            rate_limited_response = MagicMock()
            rate_limited_response.status_code = 429
            rate_limited_response.headers = {"Retry-After": "1"}

            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = elevenlabs_response

            # First call returns rate limit, second call succeeds
            service._client.post = MagicMock(side_effect=[rate_limited_response, success_response])

            with patch.object(ts_module.time, "sleep"):  # Don't actually sleep in tests
                transcript = service._transcribe_elevenlabs("https://test-url")

            assert transcript.transcription_service == "elevenlabs"
            assert service._client.post.call_count == 2

    def test_transcribe_elevenlabs_api_error(self):
        """Test ElevenLabs API error handling."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            error_response = MagicMock()
            error_response.status_code = 500
            error_response.text = "Internal Server Error"
            service._client.post = MagicMock(return_value=error_response)

            with pytest.raises(TranscriptionError, match="ElevenLabs API error"):
                service._transcribe_elevenlabs("https://test-url")


class TestAssemblyAITranscription:
    """Tests for AssemblyAI transcription."""

    def test_parse_assemblyai_response(self, assemblyai_response):
        """Test parsing AssemblyAI API response."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()
            transcript = service._parse_assemblyai_response(assemblyai_response)

            assert transcript.transcription_service == "assemblyai"
            assert len(transcript.segments) == 2

            # Check first segment
            assert transcript.segments[0].text == "Hello world"
            assert transcript.segments[0].start_time == 0.0
            assert transcript.segments[0].end_time == 1.0
            assert transcript.segments[0].speaker.id == "speaker_A"

            # Check second segment
            assert transcript.segments[1].text == "How are you"
            assert transcript.segments[1].start_time == 2.0
            assert transcript.segments[1].end_time == 2.8
            assert transcript.segments[1].speaker.id == "speaker_B"

    def test_transcribe_assemblyai_success(self, assemblyai_response):
        """Test successful AssemblyAI transcription with polling."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService(provider="assemblyai")

            # Submit response
            submit_response = MagicMock()
            submit_response.status_code = 200
            submit_response.json.return_value = {"id": "test-id"}

            # Polling response (processing, then completed)
            processing_response = MagicMock()
            processing_response.status_code = 200
            processing_response.json.return_value = {"status": "processing"}

            completed_response = MagicMock()
            completed_response.status_code = 200
            completed_response.json.return_value = assemblyai_response

            service._client.post = MagicMock(return_value=submit_response)
            service._client.get = MagicMock(side_effect=[processing_response, completed_response])

            with patch.object(ts_module.time, "sleep"):  # Don't actually sleep in tests
                transcript = service._transcribe_assemblyai("https://test-url")

            assert transcript.transcription_service == "assemblyai"
            assert service._client.get.call_count == 2

    def test_transcribe_assemblyai_error(self):
        """Test AssemblyAI error handling."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService(provider="assemblyai")

            submit_response = MagicMock()
            submit_response.status_code = 200
            submit_response.json.return_value = {"id": "test-id"}

            error_response = MagicMock()
            error_response.status_code = 200
            error_response.json.return_value = {"status": "error", "error": "Processing failed"}

            service._client.post = MagicMock(return_value=submit_response)
            service._client.get = MagicMock(return_value=error_response)

            with pytest.raises(TranscriptionError, match="AssemblyAI error"):
                service._transcribe_assemblyai("https://test-url")


class TestDeepgramTranscription:
    """Tests for Deepgram transcription."""

    def test_parse_deepgram_response(self, deepgram_response):
        """Test parsing Deepgram API response."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()
            transcript = service._parse_deepgram_response(deepgram_response)

            assert transcript.transcription_service == "deepgram"
            assert transcript.transcription_model == "nova-2"
            assert transcript.audio_duration_seconds == 300.5
            assert len(transcript.segments) == 2

            assert transcript.segments[0].text == "Hello world"
            assert transcript.segments[0].speaker.id == "speaker_0"
            assert transcript.segments[1].text == "How are you"
            assert transcript.segments[1].speaker.id == "speaker_1"

    def test_transcribe_deepgram_success(self, deepgram_response):
        """Test successful Deepgram transcription."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService(provider="deepgram")

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = deepgram_response
            service._client.post = MagicMock(return_value=mock_response)

            transcript = service._transcribe_deepgram("https://test-url")

            assert transcript.transcription_service == "deepgram"

    def test_transcribe_deepgram_api_error(self):
        """Test Deepgram API error handling."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService(provider="deepgram")

            error_response = MagicMock()
            error_response.status_code = 401
            error_response.text = "Unauthorized"
            service._client.post = MagicMock(return_value=error_response)

            with pytest.raises(TranscriptionError, match="Deepgram API error"):
                service._transcribe_deepgram("https://test-url")


class TestTranscribeMethod:
    """Tests for the main transcribe method."""

    def test_transcribe_no_s3_bucket(self, sample_source):
        """Test that transcribe raises when S3 bucket is not configured."""
        mock_settings = create_mock_settings(s3_bucket=None)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            with pytest.raises(ValueError, match="S3 bucket not configured"):
                service.transcribe(sample_source)

    def test_transcribe_no_files(self):
        """Test that transcribe raises when source has no files."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            source = Source(
                id="source:deploy:20250119_vinci_01/gopro_01",
                deployment_id="deploy:20250119_vinci_01",
                device_type=DeviceType.GOPRO,
                device_number=1,
                files=[],
            )
            service = TranscriptionService()

            with pytest.raises(TranscriptionError, match="has no files"):
                service.transcribe(source)

    def test_transcribe_single_file(self, sample_source, elevenlabs_response, temp_data_dir):
        """Test transcribing a source with a single file."""
        mock_settings = create_mock_settings(data_dir=temp_data_dir)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            # Mock S3 and API calls
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://test-url"
            service._s3_client = mock_s3

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = elevenlabs_response
            service._client.post = MagicMock(return_value=mock_response)

            transcript = service.transcribe(sample_source)

            assert transcript.source_id == sample_source.id
            assert transcript.transcription_service == "elevenlabs"
            assert len(transcript.segments) == 2

    def test_transcribe_multi_file_offset(self, sample_source_multi_file, elevenlabs_response, temp_data_dir):
        """Test that multi-file transcription correctly offsets timestamps."""
        mock_settings = create_mock_settings(data_dir=temp_data_dir)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            # Mock S3 and API calls
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://test-url"
            service._s3_client = mock_s3

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = elevenlabs_response
            service._client.post = MagicMock(return_value=mock_response)

            transcript = service.transcribe(sample_source_multi_file)

            # Should have 4 segments (2 per file)
            assert len(transcript.segments) == 4

            # First file segments should have original timestamps
            assert transcript.segments[0].start_time < 300.0
            assert transcript.segments[1].end_time < 300.0

            # Second file segments should be offset by 300 seconds
            assert transcript.segments[2].start_time >= 300.0
            assert transcript.segments[3].start_time >= 300.0


class TestTranscriptSaving:
    """Tests for transcript saving functionality."""

    def test_save_transcript(self, sample_source, temp_data_dir):
        """Test that transcripts are saved to the correct location."""
        mock_settings = create_mock_settings(data_dir=temp_data_dir)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            transcript = RawTranscript(
                id=RawTranscript.generate_id(sample_source.id),
                source_id=sample_source.id,
                language_code="en",
                transcription_service="elevenlabs",
                segments=[
                    TranscriptSegment(
                        text="Hello world",
                        start_time=0.0,
                        end_time=1.0,
                        speaker=TranscriptSpeaker(id="speaker_1"),
                    ),
                ],
            )

            saved_path = service._save_transcript(sample_source, transcript)

            assert saved_path.exists()
            assert saved_path.name == "raw_transcript.json"

            # Verify content
            saved_data = json.loads(saved_path.read_text())
            assert saved_data["source_id"] == sample_source.id
            assert len(saved_data["segments"]) == 1


class TestStatusUpdates:
    """Tests for source status update functionality."""

    def test_update_source_status(self, sample_source, temp_data_dir):
        """Test updating source transcript status."""
        mock_settings = create_mock_settings(data_dir=temp_data_dir)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            # Create source directory and file
            source_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
            source_dir.mkdir(parents=True)
            source_json = source_dir / "source.json"
            source_json.write_text(sample_source.model_dump_json(indent=2))

            service.update_source_status(sample_source, TranscriptStatus.COMPLETE)

            # Verify status was updated
            updated_data = json.loads(source_json.read_text())
            assert updated_data["transcript_status"] == "complete"

    def test_transcribe_and_update_success(self, sample_source, elevenlabs_response, temp_data_dir):
        """Test transcribe_and_update method on success."""
        mock_settings = create_mock_settings(data_dir=temp_data_dir)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            # Create source directory and file
            source_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
            source_dir.mkdir(parents=True)
            source_json = source_dir / "source.json"
            source_json.write_text(sample_source.model_dump_json(indent=2))

            # Mock S3 and API calls
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://test-url"
            service._s3_client = mock_s3

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = elevenlabs_response
            service._client.post = MagicMock(return_value=mock_response)

            transcript = service.transcribe_and_update(sample_source)

            assert transcript is not None
            # Verify final status is complete
            updated_data = json.loads(source_json.read_text())
            assert updated_data["transcript_status"] == "complete"

    def test_transcribe_and_update_failure(self, sample_source, temp_data_dir):
        """Test transcribe_and_update method on failure."""
        mock_settings = create_mock_settings(data_dir=temp_data_dir)
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            # Create source directory and file
            source_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
            source_dir.mkdir(parents=True)
            source_json = source_dir / "source.json"
            source_json.write_text(sample_source.model_dump_json(indent=2))

            # Mock S3 and API calls to fail
            mock_s3 = MagicMock()
            mock_s3.generate_presigned_url.return_value = "https://test-url"
            service._s3_client = mock_s3

            error_response = MagicMock()
            error_response.status_code = 500
            error_response.text = "Server Error"
            service._client.post = MagicMock(return_value=error_response)

            with pytest.raises(TranscriptionError):
                service.transcribe_and_update(sample_source)

            # Verify final status is failed
            updated_data = json.loads(source_json.read_text())
            assert updated_data["transcript_status"] == "failed"


class TestWordMerging:
    """Tests for word segment merging functionality."""

    def test_merge_word_segments_by_speaker(self):
        """Test that words are merged by speaker."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            segments = [
                TranscriptSegment(text="Hello", start_time=0.0, end_time=0.5, speaker=TranscriptSpeaker(id="A")),
                TranscriptSegment(text="world", start_time=0.5, end_time=1.0, speaker=TranscriptSpeaker(id="A")),
                TranscriptSegment(text="Hi", start_time=1.2, end_time=1.5, speaker=TranscriptSpeaker(id="B")),
            ]

            merged = service._merge_word_segments(segments)

            assert len(merged) == 2
            assert merged[0].text == "Hello world"
            assert merged[0].speaker.id == "A"
            assert merged[1].text == "Hi"
            assert merged[1].speaker.id == "B"

    def test_merge_word_segments_by_pause(self):
        """Test that words are split on long pauses."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            segments = [
                TranscriptSegment(text="Hello", start_time=0.0, end_time=0.5, speaker=TranscriptSpeaker(id="A")),
                TranscriptSegment(text="world", start_time=2.0, end_time=2.5, speaker=TranscriptSpeaker(id="A")),  # Long gap
            ]

            merged = service._merge_word_segments(segments, pause_threshold=1.0)

            assert len(merged) == 2
            assert merged[0].text == "Hello"
            assert merged[1].text == "world"

    def test_merge_word_segments_empty(self):
        """Test merging empty segments list."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            service = TranscriptionService()

            merged = service._merge_word_segments([])

            assert merged == []


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self):
        """Test that service works as context manager."""
        mock_settings = create_mock_settings()
        with patch.object(ts_module, "get_settings", return_value=mock_settings):
            with TranscriptionService() as service:
                assert service is not None
                assert service._client is not None

            # Client should be closed after exiting context
            assert service._client.is_closed
