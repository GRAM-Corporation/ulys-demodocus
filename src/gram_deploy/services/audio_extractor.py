"""Audio Extractor - isolates audio tracks from video files for transcription.

Responsible for:
- Extracting audio from video files using ffmpeg
- Caching extracted audio to avoid re-processing
- Computing audio fingerprints for time alignment
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Optional

from gram_deploy.models import Source


class AudioExtractor:
    """Extracts audio from video files for transcription and analysis."""

    def __init__(self, cache_dir: str):
        """Initialize the extractor with a cache directory.

        Args:
            cache_dir: Directory for caching extracted audio files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(
        self,
        video_path: str,
        output_format: str = "wav",
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> str:
        """Extract audio track from a video file.

        Args:
            video_path: Path to the video file
            output_format: Output format (wav, mp3, flac)
            sample_rate: Audio sample rate in Hz (16000 optimal for speech)
            mono: Whether to convert to mono

        Returns:
            Path to the extracted audio file
        """
        video_path = Path(video_path)

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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        return str(output_path)

    def extract_for_source(self, source: Source) -> list[str]:
        """Extract audio from all files in a source.

        Args:
            source: The Source entity

        Returns:
            Paths to extracted audio files, in same order as source files
        """
        audio_paths = []
        for source_file in source.files:
            audio_path = self.extract_audio(source_file.file_path)
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

        hasher = hashlib.md5()
        with open(path, "rb") as f:
            # Hash first and last chunks for speed on large files
            hasher.update(f.read(chunk_size))
            f.seek(-chunk_size, 2)  # Seek to end - chunk_size
            hasher.update(f.read(chunk_size))
            # Include file size
            hasher.update(str(path.stat().st_size).encode())

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
