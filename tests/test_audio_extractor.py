"""Tests for the AudioExtractor service.

Tests cover:
- Audio extraction from video files
- Video metadata extraction with ffprobe
- Source audio extraction with concatenation
- Error handling for missing ffmpeg/ffprobe
- Handling corrupt/missing video files
- Output validation
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from gram_deploy.services.audio_extractor import (
    AudioExtractor,
    AudioExtractorError,
    AudioExtractionError,
    FFmpegNotFoundError,
    VideoFileError,
)


class TestAudioExtractorInit:
    """Tests for AudioExtractor initialization."""

    def test_init_with_cache_dir(self, tmp_path):
        """Should create cache directory if it doesn't exist."""
        cache_dir = tmp_path / "audio_cache"
        extractor = AudioExtractor(cache_dir=str(cache_dir))
        assert cache_dir.exists()
        assert extractor.cache_dir == cache_dir

    def test_init_without_cache_dir(self):
        """Should work without cache directory for basic operations."""
        extractor = AudioExtractor()
        assert extractor.cache_dir is None


class TestFFmpegAvailability:
    """Tests for ffmpeg/ffprobe availability checks."""

    def test_check_ffmpeg_available_success(self):
        """Should return True when ffmpeg is available."""
        extractor = AudioExtractor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extractor.check_ffmpeg_available()
            assert result is True
            mock_run.assert_called_once()

    def test_check_ffmpeg_available_not_found(self):
        """Should return False when ffmpeg is not found."""
        extractor = AudioExtractor()

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = extractor.check_ffmpeg_available()
            assert result is False

    def test_check_ffmpeg_available_cached(self):
        """Should cache the result after first check."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        with patch("subprocess.run") as mock_run:
            result = extractor.check_ffmpeg_available()
            assert result is True
            mock_run.assert_not_called()

    def test_check_ffprobe_available_success(self):
        """Should return True when ffprobe is available."""
        extractor = AudioExtractor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extractor.check_ffprobe_available()
            assert result is True

    def test_check_ffprobe_available_not_found(self):
        """Should return False when ffprobe is not found."""
        extractor = AudioExtractor()

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = extractor.check_ffprobe_available()
            assert result is False

    def test_require_ffmpeg_raises_when_not_available(self):
        """Should raise FFmpegNotFoundError when ffmpeg is not available."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = False

        with pytest.raises(FFmpegNotFoundError) as exc_info:
            extractor._require_ffmpeg()
        assert "ffmpeg is not installed" in str(exc_info.value)

    def test_require_ffprobe_raises_when_not_available(self):
        """Should raise FFmpegNotFoundError when ffprobe is not available."""
        extractor = AudioExtractor()
        extractor._ffprobe_available = False

        with pytest.raises(FFmpegNotFoundError) as exc_info:
            extractor._require_ffprobe()
        assert "ffprobe is not installed" in str(exc_info.value)


class TestExtractAudio:
    """Tests for extract_audio method."""

    def test_extract_audio_success(self, tmp_path):
        """Should successfully extract audio from video."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        # Create a fake video file
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")
        output_path = tmp_path / "output.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Create the output file to simulate ffmpeg creating it
            output_path.write_bytes(b"fake audio content")

            result = extractor.extract_audio(str(video_path), str(output_path))

            assert result == output_path
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-vn" in cmd
            assert "-acodec" in cmd
            assert "pcm_s16le" in cmd
            assert "-ar" in cmd
            assert "16000" in cmd
            assert "-ac" in cmd
            assert "1" in cmd

    def test_extract_audio_video_not_found(self, tmp_path):
        """Should raise VideoFileError when video file doesn't exist."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        video_path = tmp_path / "nonexistent.mp4"
        output_path = tmp_path / "output.wav"

        with pytest.raises(VideoFileError) as exc_info:
            extractor.extract_audio(str(video_path), str(output_path))
        assert "not found" in str(exc_info.value)

    def test_extract_audio_empty_video(self, tmp_path):
        """Should raise VideoFileError when video file is empty."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        video_path = tmp_path / "empty.mp4"
        video_path.touch()  # Create empty file
        output_path = tmp_path / "output.wav"

        with pytest.raises(VideoFileError) as exc_info:
            extractor.extract_audio(str(video_path), str(output_path))
        assert "empty" in str(exc_info.value)

    def test_extract_audio_ffmpeg_failure(self, tmp_path):
        """Should raise AudioExtractionError when ffmpeg fails."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")
        output_path = tmp_path / "output.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Error: corrupt file"
            )
            with pytest.raises(AudioExtractionError) as exc_info:
                extractor.extract_audio(str(video_path), str(output_path))
            assert "ffmpeg failed" in str(exc_info.value)

    def test_extract_audio_timeout(self, tmp_path):
        """Should raise AudioExtractionError on timeout."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")
        output_path = tmp_path / "output.wav"

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 600)):
            with pytest.raises(AudioExtractionError) as exc_info:
                extractor.extract_audio(str(video_path), str(output_path))
            assert "timed out" in str(exc_info.value)

    def test_extract_audio_output_validation(self, tmp_path):
        """Should raise AudioExtractionError if output is empty."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")
        output_path = tmp_path / "output.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Create empty output file
            output_path.touch()

            with pytest.raises(AudioExtractionError) as exc_info:
                extractor.extract_audio(str(video_path), str(output_path))
            assert "empty" in str(exc_info.value)


class TestGetVideoMetadata:
    """Tests for get_video_metadata method."""

    SAMPLE_FFPROBE_OUTPUT = {
        "format": {
            "duration": "120.5"
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001"
            },
            {
                "codec_type": "audio",
                "codec_name": "aac"
            }
        ]
    }

    def test_get_video_metadata_success(self, tmp_path):
        """Should parse ffprobe output correctly."""
        extractor = AudioExtractor()
        extractor._ffprobe_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(self.SAMPLE_FFPROBE_OUTPUT)
            )

            metadata = extractor.get_video_metadata(str(video_path))

            assert metadata["duration_seconds"] == 120.5
            assert metadata["video_codec"] == "h264"
            assert metadata["audio_codec"] == "aac"
            assert metadata["resolution"] == "1920x1080"
            assert metadata["fps"] == 29.97  # 30000/1001 rounded
            assert metadata["file_size_bytes"] == len(b"fake video content")

    def test_get_video_metadata_missing_fields(self, tmp_path):
        """Should handle missing fields gracefully."""
        extractor = AudioExtractor()
        extractor._ffprobe_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        minimal_output = {"format": {}, "streams": []}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(minimal_output)
            )

            metadata = extractor.get_video_metadata(str(video_path))

            assert metadata["duration_seconds"] == 0.0
            assert metadata["video_codec"] is None
            assert metadata["audio_codec"] is None
            assert metadata["resolution"] is None
            assert metadata["fps"] is None

    def test_get_video_metadata_ffprobe_failure(self, tmp_path):
        """Should raise VideoFileError when ffprobe fails."""
        extractor = AudioExtractor()
        extractor._ffprobe_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Error reading file"
            )
            with pytest.raises(VideoFileError) as exc_info:
                extractor.get_video_metadata(str(video_path))
            assert "ffprobe failed" in str(exc_info.value)

    def test_get_video_metadata_invalid_json(self, tmp_path):
        """Should raise VideoFileError on invalid JSON output."""
        extractor = AudioExtractor()
        extractor._ffprobe_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not valid json"
            )
            with pytest.raises(VideoFileError) as exc_info:
                extractor.get_video_metadata(str(video_path))
            assert "invalid JSON" in str(exc_info.value)

    def test_get_video_metadata_file_not_found(self, tmp_path):
        """Should raise VideoFileError when file doesn't exist."""
        extractor = AudioExtractor()
        extractor._ffprobe_available = True

        video_path = tmp_path / "nonexistent.mp4"

        with pytest.raises(VideoFileError) as exc_info:
            extractor.get_video_metadata(str(video_path))
        assert "not found" in str(exc_info.value)


class TestExtractSourceAudio:
    """Tests for extract_source_audio method."""

    def test_extract_source_audio_single_file(self, tmp_path):
        """Should extract audio from single file source."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        # Create source directory with source.json
        source_dir = tmp_path / "source_gopro_01"
        source_dir.mkdir()

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        source_json = {
            "files": [{"file_path": str(video_path)}]
        }
        (source_dir / "source.json").write_text(json.dumps(source_json))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Create output to simulate ffmpeg
            (source_dir / "audio.wav").write_bytes(b"fake audio")

            result = extractor.extract_source_audio(str(source_dir))

            assert result == source_dir / "audio.wav"

    def test_extract_source_audio_multiple_files(self, tmp_path):
        """Should extract and concatenate audio from multiple files."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        source_dir = tmp_path / "source_gopro_01"
        source_dir.mkdir()

        # Create multiple video files
        video1 = tmp_path / "video1.mp4"
        video2 = tmp_path / "video2.mp4"
        video1.write_bytes(b"fake video content 1")
        video2.write_bytes(b"fake video content 2")

        source_json = {
            "files": [
                {"file_path": str(video1)},
                {"file_path": str(video2)},
            ]
        }
        (source_dir / "source.json").write_text(json.dumps(source_json))

        call_count = 0
        def mock_run_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Create temp files for individual extractions
            if call_count <= 2:
                temp_dir = source_dir / ".temp_audio"
                temp_dir.mkdir(exist_ok=True)
                (temp_dir / f"part_{call_count-1:04d}.wav").write_bytes(b"audio")
            elif call_count == 3:
                # Concatenation call
                (source_dir / "audio.wav").write_bytes(b"concatenated audio")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            result = extractor.extract_source_audio(str(source_dir))

            assert result == source_dir / "audio.wav"
            # Should have called ffmpeg 3 times: 2 extractions + 1 concat
            assert call_count == 3

    def test_extract_source_audio_no_source_json(self, tmp_path):
        """Should raise FileNotFoundError when source.json is missing."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        source_dir = tmp_path / "source_gopro_01"
        source_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            extractor.extract_source_audio(str(source_dir))
        assert "source.json not found" in str(exc_info.value)

    def test_extract_source_audio_no_files(self, tmp_path):
        """Should raise VideoFileError when files list is empty."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        source_dir = tmp_path / "source_gopro_01"
        source_dir.mkdir()
        (source_dir / "source.json").write_text(json.dumps({"files": []}))

        with pytest.raises(VideoFileError) as exc_info:
            extractor.extract_source_audio(str(source_dir))
        assert "No files found" in str(exc_info.value)

    def test_extract_source_audio_missing_file_path(self, tmp_path):
        """Should raise VideoFileError when file_path is missing."""
        extractor = AudioExtractor()
        extractor._ffmpeg_available = True

        source_dir = tmp_path / "source_gopro_01"
        source_dir.mkdir()
        (source_dir / "source.json").write_text(json.dumps({
            "files": [{"filename": "video.mp4"}]  # missing file_path
        }))

        with pytest.raises(VideoFileError) as exc_info:
            extractor.extract_source_audio(str(source_dir))
        assert "file_path missing" in str(exc_info.value)


class TestExtractAudioCached:
    """Tests for extract_audio_cached method."""

    def test_extract_audio_cached_returns_cached(self, tmp_path):
        """Should return cached file if it exists."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        extractor = AudioExtractor(cache_dir=str(cache_dir))
        extractor._ffmpeg_available = True

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        # Pre-create a cached file
        cached_file = cache_dir / f"{extractor._compute_file_hash(str(video_path))}.wav"
        cached_file.write_bytes(b"cached audio")

        with patch("subprocess.run") as mock_run:
            result = extractor.extract_audio_cached(str(video_path))
            mock_run.assert_not_called()
            assert result == str(cached_file)

    def test_extract_audio_cached_no_cache_dir_raises(self):
        """Should raise ValueError if cache_dir not set."""
        extractor = AudioExtractor()  # No cache_dir

        with pytest.raises(ValueError) as exc_info:
            extractor.extract_audio_cached("/some/video.mp4")
        assert "cache_dir must be set" in str(exc_info.value)


class TestFileHash:
    """Tests for _compute_file_hash method."""

    def test_compute_file_hash_consistent(self, tmp_path):
        """Should produce consistent hashes for same file."""
        extractor = AudioExtractor()

        file_path = tmp_path / "test.mp4"
        file_path.write_bytes(b"x" * 20000)  # Larger than chunk size

        hash1 = extractor._compute_file_hash(str(file_path))
        hash2 = extractor._compute_file_hash(str(file_path))

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest length

    def test_compute_file_hash_different_files(self, tmp_path):
        """Should produce different hashes for different files."""
        extractor = AudioExtractor()

        file1 = tmp_path / "test1.mp4"
        file2 = tmp_path / "test2.mp4"
        file1.write_bytes(b"content1" + b"x" * 20000)
        file2.write_bytes(b"content2" + b"x" * 20000)

        hash1 = extractor._compute_file_hash(str(file1))
        hash2 = extractor._compute_file_hash(str(file2))

        assert hash1 != hash2

    def test_compute_file_hash_nonexistent(self, tmp_path):
        """Should return hash of path string for nonexistent files."""
        extractor = AudioExtractor()

        file_path = tmp_path / "nonexistent.mp4"
        result = extractor._compute_file_hash(str(file_path))

        assert len(result) == 32
