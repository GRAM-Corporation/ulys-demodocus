"""Audio Extractor - isolates audio tracks from video files for transcription.

Responsible for:
- Extracting audio from video files using ffmpeg
- Caching extracted audio to avoid re-processing
- Computing audio fingerprints for time alignment
- Extracting video metadata using ffprobe
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional

from gram_deploy.models import Source


class AudioExtractorError(Exception):
    """Base exception for audio extraction errors."""
    pass


class FFmpegNotFoundError(AudioExtractorError):
    """Raised when ffmpeg or ffprobe is not available."""
    pass


class VideoFileError(AudioExtractorError):
    """Raised when video file is missing or corrupt."""
    pass


class AudioExtractionError(AudioExtractorError):
    """Raised when audio extraction fails."""
    pass


class AudioExtractor:
    """Extracts audio from video files for transcription and analysis."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the extractor with an optional cache directory.

        Args:
            cache_dir: Directory for caching extracted audio files (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_available: Optional[bool] = None
        self._ffprobe_available: Optional[bool] = None

    def check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available in the system PATH.

        Returns:
            True if ffmpeg is available, False otherwise
        """
        if self._ffmpeg_available is not None:
            return self._ffmpeg_available

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=10
            )
            self._ffmpeg_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._ffmpeg_available = False

        return self._ffmpeg_available

    def check_ffprobe_available(self) -> bool:
        """Check if ffprobe is available in the system PATH.

        Returns:
            True if ffprobe is available, False otherwise
        """
        if self._ffprobe_available is not None:
            return self._ffprobe_available

        try:
            result = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                timeout=10
            )
            self._ffprobe_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._ffprobe_available = False

        return self._ffprobe_available

    def _require_ffmpeg(self) -> None:
        """Raise an error if ffmpeg is not available."""
        if not self.check_ffmpeg_available():
            raise FFmpegNotFoundError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg to use audio extraction features."
            )

    def _require_ffprobe(self) -> None:
        """Raise an error if ffprobe is not available."""
        if not self.check_ffprobe_available():
            raise FFmpegNotFoundError(
                "ffprobe is not installed or not in PATH. "
                "Please install ffmpeg (which includes ffprobe) to use metadata extraction."
            )

    def _validate_video_file(self, video_path: Path) -> None:
        """Validate that a video file exists and is readable.

        Args:
            video_path: Path to the video file

        Raises:
            VideoFileError: If the file doesn't exist or is empty
        """
        if not video_path.exists():
            raise VideoFileError(f"Video file not found: {video_path}")
        if not video_path.is_file():
            raise VideoFileError(f"Not a file: {video_path}")
        if video_path.stat().st_size == 0:
            raise VideoFileError(f"Video file is empty: {video_path}")

    def _validate_output_file(self, output_path: Path) -> None:
        """Validate that an output file was created successfully.

        Args:
            output_path: Path to the output file

        Raises:
            AudioExtractionError: If the file doesn't exist or is empty
        """
        if not output_path.exists():
            raise AudioExtractionError(f"Output file was not created: {output_path}")
        if output_path.stat().st_size == 0:
            raise AudioExtractionError(f"Output file is empty: {output_path}")

    def extract_audio(self, video_path: str, output_path: str) -> Path:
        """Extract audio track from a video file.

        Outputs WAV format, 16kHz mono, optimal for speech recognition.
        Uses command: ffmpeg -i {video} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output}

        Args:
            video_path: Path to the video file
            output_path: Path for the output audio file

        Returns:
            Path to the extracted audio file

        Raises:
            FFmpegNotFoundError: If ffmpeg is not available
            VideoFileError: If video file is missing or corrupt
            AudioExtractionError: If extraction fails
        """
        self._require_ffmpeg()

        video_path = Path(video_path)
        output_path = Path(output_path)

        self._validate_video_file(video_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command - WAV, 16kHz mono for speech recognition
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",  # Overwrite
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                raise AudioExtractionError(
                    f"ffmpeg failed with code {result.returncode}: {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise AudioExtractionError(
                f"Audio extraction timed out for {video_path}"
            )

        self._validate_output_file(output_path)

        return output_path

    def extract_audio_cached(
        self,
        video_path: str,
        output_format: str = "wav",
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> str:
        """Extract audio track from a video file with caching.

        Args:
            video_path: Path to the video file
            output_format: Output format (wav, mp3, flac)
            sample_rate: Audio sample rate in Hz (16000 optimal for speech)
            mono: Whether to convert to mono

        Returns:
            Path to the extracted audio file

        Raises:
            ValueError: If cache_dir was not set during initialization
        """
        if not self.cache_dir:
            raise ValueError("cache_dir must be set to use cached extraction")

        self._require_ffmpeg()

        video_path_obj = Path(video_path)
        self._validate_video_file(video_path_obj)

        # Generate cache key from file content hash
        cache_key = self._compute_file_hash(str(video_path))
        output_filename = f"{cache_key}.{output_format}"
        output_path = self.cache_dir / output_filename

        # Return cached version if exists
        if output_path.exists():
            return str(output_path)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", self._get_codec(output_format),
            "-ar", str(sample_rate),
        ]

        if mono:
            cmd.extend(["-ac", "1"])

        # Add format-specific options
        if output_format == "wav":
            cmd.extend(["-f", "wav"])
        elif output_format == "mp3":
            cmd.extend(["-b:a", "128k"])

        # Normalization for better speech recognition
        cmd.extend([
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-y",  # Overwrite
            str(output_path),
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise AudioExtractionError(f"ffmpeg failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioExtractionError(f"Audio extraction timed out for {video_path}")

        self._validate_output_file(output_path)

        return str(output_path)

    def get_video_metadata(self, video_path: str) -> dict:
        """Extract metadata from a video file using ffprobe.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with:
                - duration_seconds: float
                - video_codec: str or None
                - audio_codec: str or None
                - resolution: str (e.g., "1920x1080") or None
                - fps: float or None
                - file_size_bytes: int

        Raises:
            FFmpegNotFoundError: If ffprobe is not available
            VideoFileError: If video file is missing or corrupt
        """
        self._require_ffprobe()

        video_path = Path(video_path)
        self._validate_video_file(video_path)

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise VideoFileError(
                    f"ffprobe failed to read video: {result.stderr}"
                )

            data = json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise VideoFileError(f"ffprobe timed out for {video_path}")
        except json.JSONDecodeError as e:
            raise VideoFileError(f"ffprobe returned invalid JSON: {e}")

        metadata = {
            "duration_seconds": 0.0,
            "video_codec": None,
            "audio_codec": None,
            "resolution": None,
            "fps": None,
            "file_size_bytes": video_path.stat().st_size,
        }

        # Extract duration from format
        if "format" in data:
            duration_str = data["format"].get("duration")
            if duration_str:
                try:
                    metadata["duration_seconds"] = float(duration_str)
                except ValueError:
                    pass

        # Extract stream info
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type")

            if codec_type == "video":
                metadata["video_codec"] = stream.get("codec_name")
                width = stream.get("width")
                height = stream.get("height")
                if width and height:
                    metadata["resolution"] = f"{width}x{height}"

                # Parse fps from r_frame_rate (e.g., "30000/1001")
                fps_str = stream.get("r_frame_rate", "")
                if "/" in fps_str:
                    try:
                        num, den = fps_str.split("/")
                        if int(den) > 0:
                            metadata["fps"] = round(int(num) / int(den), 2)
                    except (ValueError, ZeroDivisionError):
                        pass

            elif codec_type == "audio":
                metadata["audio_codec"] = stream.get("codec_name")

        return metadata

    def extract_source_audio(self, source_path: str) -> Path:
        """Extract and concatenate audio from all video files in a source.

        Reads source.json to get the file list, extracts audio from each
        video file, concatenates them if multiple, and writes to audio.wav.

        Args:
            source_path: Path to the source directory (contains source.json)

        Returns:
            Path to the extracted audio file ({source_path}/audio.wav)

        Raises:
            FileNotFoundError: If source.json not found
            FFmpegNotFoundError: If ffmpeg is not available
            VideoFileError: If video files are missing or corrupt
            AudioExtractionError: If extraction fails
        """
        self._require_ffmpeg()

        source_path = Path(source_path)
        source_json_path = source_path / "source.json"

        if not source_json_path.exists():
            raise FileNotFoundError(f"source.json not found in {source_path}")

        # Read source.json to get file list
        with open(source_json_path, "r") as f:
            source_data = json.load(f)

        files = source_data.get("files", [])
        if not files:
            raise VideoFileError("No files found in source.json")

        output_path = source_path / "audio.wav"

        if len(files) == 1:
            # Single file - extract directly
            video_file = files[0].get("file_path")
            if not video_file:
                raise VideoFileError("file_path missing in source.json")
            return self.extract_audio(video_file, str(output_path))

        # Multiple files - extract each then concatenate
        temp_audio_files = []
        temp_dir = source_path / ".temp_audio"
        temp_dir.mkdir(exist_ok=True)

        try:
            for i, file_info in enumerate(files):
                video_file = file_info.get("file_path")
                if not video_file:
                    raise VideoFileError(f"file_path missing for file {i} in source.json")

                temp_audio = temp_dir / f"part_{i:04d}.wav"
                self.extract_audio(video_file, str(temp_audio))
                temp_audio_files.append(temp_audio)

            # Create concat list file for ffmpeg
            concat_list_path = temp_dir / "concat.txt"
            with open(concat_list_path, "w") as f:
                for audio_file in temp_audio_files:
                    f.write(f"file '{audio_file}'\n")

            # Concatenate using ffmpeg
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
                "-y",
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                raise AudioExtractionError(
                    f"Audio concatenation failed: {result.stderr}"
                )

            self._validate_output_file(output_path)

        finally:
            # Clean up temp files
            for temp_file in temp_audio_files:
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            try:
                (temp_dir / "concat.txt").unlink()
            except OSError:
                pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass

        return output_path

    def extract_for_source(self, source: Source) -> list[str]:
        """Extract audio from all files in a source using cached extraction.

        Args:
            source: The Source entity

        Returns:
            Paths to extracted audio files, in same order as source files

        Raises:
            ValueError: If cache_dir was not set during initialization
        """
        audio_paths = []
        for source_file in source.files:
            audio_path = self.extract_audio_cached(source_file.file_path)
            audio_paths.append(audio_path)
        return audio_paths

    def get_audio_fingerprint(
        self,
        audio_path: str,
        duration_seconds: float = 30.0,
        offset_seconds: float = 0.0,
    ) -> bytes:
        """Compute an audio fingerprint for time alignment.

        Uses chromaprint/fpcalc for robust audio fingerprinting.

        Args:
            audio_path: Path to the audio file
            duration_seconds: Duration to fingerprint
            offset_seconds: Offset from start to begin fingerprinting

        Returns:
            Raw fingerprint bytes
        """
        try:
            cmd = [
                "fpcalc",
                "-raw",
                "-length", str(int(duration_seconds)),
                "-offset", str(int(offset_seconds)),
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                # Fallback: compute simple hash of audio samples
                return self._compute_simple_fingerprint(audio_path, duration_seconds)

            # Parse fpcalc output
            for line in result.stdout.split("\n"):
                if line.startswith("FINGERPRINT="):
                    fp_data = line.replace("FINGERPRINT=", "")
                    return fp_data.encode()

            return b""

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._compute_simple_fingerprint(audio_path, duration_seconds)

    def _get_codec(self, output_format: str) -> str:
        """Get the ffmpeg codec for an output format."""
        codecs = {
            "wav": "pcm_s16le",
            "mp3": "libmp3lame",
            "flac": "flac",
        }
        return codecs.get(output_format, "pcm_s16le")

    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Compute a hash of a file's contents."""
        path = Path(file_path)
        if not path.exists():
            return hashlib.md5(file_path.encode()).hexdigest()

        file_size = path.stat().st_size
        hasher = hashlib.md5()

        with open(path, "rb") as f:
            # Hash first chunk
            hasher.update(f.read(chunk_size))

            # Hash last chunk if file is larger than chunk_size
            if file_size > chunk_size:
                f.seek(-chunk_size, 2)  # Seek to end - chunk_size
                hasher.update(f.read(chunk_size))

            # Include file size
            hasher.update(str(file_size).encode())

        return hasher.hexdigest()

    def _compute_simple_fingerprint(
        self,
        audio_path: str,
        duration_seconds: float,
    ) -> bytes:
        """Fallback fingerprint using audio samples."""
        try:
            # Extract raw samples with ffmpeg
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-t", str(duration_seconds),
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "8000",
                "-ac", "1",
                "pipe:1",
            ]
            result = subprocess.run(
                cmd, capture_output=True, timeout=60
            )
            return hashlib.md5(result.stdout).digest()
        except Exception:
            return b""
