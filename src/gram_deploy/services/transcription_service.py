"""Transcription Service - speech-to-text with speaker diarization.

Pure S3 presigned URL flow - no local audio extraction. Transcription providers
fetch video directly from S3 using presigned URLs.

Flow:
    Google Drive → S3 (rclone sync) → Presigned URL → Provider fetches & transcribes

Responsible for:
- Generating S3 presigned URLs for video files
- Submitting URLs to transcription APIs (ElevenLabs, AssemblyAI, Deepgram)
- Parsing responses into RawTranscript format
- Saving transcripts and updating source status
"""

import time
from pathlib import Path
from typing import Any, Optional

import boto3
import httpx
from botocore.config import Config as BotoConfig

from gram_deploy.config import get_settings, TranscriptionProvider
from gram_deploy.models import (
    RawTranscript,
    Source,
    TranscriptSegment,
    TranscriptSpeaker,
    TranscriptStatus,
    WordTiming,
)


class TranscriptionError(Exception):
    """Error during transcription."""

    pass


class TranscriptionService:
    """Handles speech-to-text transcription with speaker diarization via S3 presigned URLs.

    No local audio extraction required - transcription services fetch video directly from S3.
    """

    PROVIDERS = ("elevenlabs", "assemblyai", "deepgram")

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        data_dir: str | Path | None = None,
    ):
        """Initialize the transcription service.

        Args:
            provider: Transcription provider (elevenlabs, assemblyai, deepgram).
                     Defaults to settings.transcription_provider.
            api_key: API key for the provider. Defaults to the key from settings.
            data_dir: Base directory for deployment data. Defaults to settings.data_dir.
        """
        settings = get_settings()

        # Set provider
        if provider is None:
            self.provider = settings.transcription_provider.value
        else:
            if provider not in self.PROVIDERS:
                raise ValueError(f"Unknown provider: {provider}. Must be one of {self.PROVIDERS}")
            self.provider = provider

        # Set API key
        if api_key is None:
            self.api_key = settings.get_transcription_api_key()
        else:
            self.api_key = api_key

        # Set data directory
        self.data_dir = Path(data_dir) if data_dir else settings.data_dir

        # S3 configuration
        self.s3_bucket = settings.s3_bucket
        self.s3_region = settings.s3_region

        # Initialize S3 client lazily
        self._s3_client: Any = None

        # HTTP client for API calls
        self._client = httpx.Client(timeout=300.0)

    @property
    def s3_client(self):
        """Lazily initialize S3 client."""
        if self._s3_client is None:
            boto_config = BotoConfig(
                region_name=self.s3_region,
                signature_version="s3v4",
            )
            self._s3_client = boto3.client("s3", config=boto_config)
        return self._s3_client

    def transcribe(
        self,
        source: Source,
        provider: str | None = None,
        language: str = "en",
        local_base_path: str | Path | None = None,
    ) -> RawTranscript:
        """Transcribe a source using local files or S3 presigned URLs.

        Args:
            source: The Source entity to transcribe
            provider: Override the default provider for this transcription
            language: Language code (default: en)
            local_base_path: Base path for local video files (skips S3 if provided)

        Returns:
            RawTranscript with segments and speaker identification

        Raises:
            TranscriptionError: If transcription fails
            ValueError: If neither local path nor S3 bucket is configured
        """
        use_local = local_base_path is not None
        if not use_local and self.s3_bucket is None:
            raise ValueError("Either local_base_path or GRAM_S3_BUCKET must be configured.")

        use_provider = provider or self.provider

        # Get S3 key and generate presigned URL
        # For sources with multiple files, we transcribe the first video file
        # (in production, these would be concatenated or processed separately)
        if not source.files:
            raise TranscriptionError(f"Source {source.id} has no files")

        # Transcribe each file and merge
        all_segments: list[TranscriptSegment] = []
        current_offset = 0.0
        audio_duration = 0.0

        for source_file in source.files:
            # Get file path or URL
            if use_local:
                file_path = Path(local_base_path) / source_file.filename
                if not file_path.exists():
                    raise TranscriptionError(f"Local file not found: {file_path}")

                # Call provider-specific transcription with local file
                if use_provider == "elevenlabs":
                    file_transcript = self._transcribe_elevenlabs_file(file_path, language)
                else:
                    raise NotImplementedError(f"Local file upload not yet implemented for {use_provider}")
            else:
                s3_key = self._get_s3_key(source, source_file.filename)
                presigned_url = self._generate_presigned_url(s3_key)

                # Call provider-specific transcription with URL
                if use_provider == "elevenlabs":
                    file_transcript = self._transcribe_elevenlabs(presigned_url, language)
                elif use_provider == "assemblyai":
                    file_transcript = self._transcribe_assemblyai(presigned_url, language)
                elif use_provider == "deepgram":
                    file_transcript = self._transcribe_deepgram(presigned_url, language)
                else:
                    raise NotImplementedError(f"Provider {use_provider} not yet implemented")

            # Adjust timestamps for file position within source
            for segment in file_transcript.segments:
                adjusted_segment = TranscriptSegment(
                    text=segment.text,
                    start_time=segment.start_time + current_offset,
                    end_time=segment.end_time + current_offset,
                    speaker=segment.speaker,
                    confidence=segment.confidence,
                    words=[
                        WordTiming(
                            text=w.text,
                            start_time=w.start_time + current_offset,
                            end_time=w.end_time + current_offset,
                            confidence=w.confidence,
                        )
                        for w in (segment.words or [])
                    ]
                    if segment.words
                    else None,
                )
                all_segments.append(adjusted_segment)

            current_offset += source_file.duration_seconds
            audio_duration += file_transcript.audio_duration_seconds or source_file.duration_seconds

        # Create merged transcript
        transcript = RawTranscript(
            id=RawTranscript.generate_id(source.id),
            source_id=source.id,
            language_code=language,
            transcription_service=use_provider,
            segments=all_segments,
            audio_duration_seconds=audio_duration,
            word_count=sum(
                len(s.words) if s.words else len(s.text.split()) for s in all_segments
            ),
        )

        # Save transcript
        self._save_transcript(source, transcript)

        return transcript

    def _get_s3_key(self, source: Source, filename: str) -> str:
        """Map source to S3 bucket path.

        Convention: deployments/{deployment_id}/sources/{source_name}/{filename}

        Args:
            source: The Source entity
            filename: The video filename

        Returns:
            S3 key for the video file
        """
        # Extract deployment ID (e.g., "deploy:20250119_vinci_01")
        deployment_id = source.deployment_id

        # Convert to directory format: deploy_20250119_vinci_01
        deployment_dir = deployment_id.replace(":", "_")

        # Extract source name from source ID
        # source:deploy:20250119_vinci_01/gopro_01 -> gopro_01
        source_name = source.id.split("/")[-1]

        return f"deployments/{deployment_dir}/sources/{source_name}/{filename}"

    def _generate_presigned_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for an S3 object.

        Args:
            s3_key: The S3 object key
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL for the S3 object
        """
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.s3_bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )

    def _transcribe_elevenlabs_file(self, file_path: Path, language: str = "en") -> RawTranscript:
        """Transcribe using ElevenLabs Scribe API with direct file upload.

        Args:
            file_path: Path to the local video/audio file
            language: Language code

        Returns:
            RawTranscript with transcription results
        """
        api_url = "https://api.elevenlabs.io/v1/speech-to-text"

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "video/mp4")}
            data = {
                "model_id": "scribe_v1",
                "language_code": language,
                "diarize": "true",
                "timestamps_granularity": "word",
            }

            response = self._client.post(
                api_url,
                headers={"xi-api-key": self.api_key},
                files=files,
                data=data,
                timeout=600.0,  # 10 min timeout for large files
            )

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self._transcribe_elevenlabs_file(file_path, language)

        if response.status_code != 200:
            raise TranscriptionError(
                f"ElevenLabs API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return self._parse_elevenlabs_response(result)

    def _transcribe_elevenlabs(self, url: str, language: str = "en") -> RawTranscript:
        """Transcribe using ElevenLabs Scribe API with URL.

        Args:
            url: Presigned URL to the video file
            language: Language code

        Returns:
            RawTranscript with transcription results
        """
        api_url = "https://api.elevenlabs.io/v1/speech-to-text"

        response = self._client.post(
            api_url,
            headers={"xi-api-key": self.api_key},
            json={
                "url": url,
                "diarization": True,
                "timestamps": True,
                "model_id": "scribe_v1",
                "language_code": language,
            },
        )

        if response.status_code == 429:
            # Rate limited - wait and retry
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self._transcribe_elevenlabs(url, language)

        if response.status_code != 200:
            raise TranscriptionError(
                f"ElevenLabs API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return self._parse_elevenlabs_response(result)

    def _parse_elevenlabs_response(self, response: dict) -> RawTranscript:
        """Parse ElevenLabs API response into RawTranscript format."""
        segments: list[TranscriptSegment] = []

        # ElevenLabs returns word-level data with speaker info
        for word_data in response.get("words", []):
            speaker_id = word_data.get("speaker_id")
            segment = TranscriptSegment(
                text=word_data.get("text", ""),
                start_time=word_data.get("start", 0.0),
                end_time=word_data.get("end", 0.0),
                speaker=TranscriptSpeaker(id=speaker_id) if speaker_id else None,
                confidence=word_data.get("confidence"),
            )
            segments.append(segment)

        # Merge adjacent words from same speaker into segments
        merged_segments = self._merge_word_segments(segments)

        return RawTranscript(
            id="transcript:source:temp_elevenlabs",
            source_id="",  # Will be set by caller
            language_code=response.get("language_code", "en"),
            transcription_service="elevenlabs",
            transcription_model="scribe_v1",
            segments=merged_segments,
            audio_duration_seconds=response.get("audio_duration"),
        )

    def _transcribe_assemblyai(self, url: str, language: str = "en") -> RawTranscript:
        """Transcribe using AssemblyAI API with URL.

        Args:
            url: Presigned URL to the video file
            language: Language code

        Returns:
            RawTranscript with transcription results
        """
        # Submit transcription request
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        transcript_request = {
            "audio_url": url,
            "language_code": language,
            "speaker_labels": True,
        }

        response = self._client.post(
            transcript_url,
            headers={"authorization": self.api_key},
            json=transcript_request,
        )

        if response.status_code != 200:
            raise TranscriptionError(
                f"AssemblyAI API error: {response.status_code} - {response.text}"
            )

        transcript_id = response.json()["id"]

        # Poll for completion
        polling_url = f"{transcript_url}/{transcript_id}"
        max_attempts = 120  # 10 minutes with 5s intervals
        attempts = 0

        while attempts < max_attempts:
            poll_response = self._client.get(
                polling_url,
                headers={"authorization": self.api_key},
            )

            if poll_response.status_code != 200:
                raise TranscriptionError(
                    f"AssemblyAI polling error: {poll_response.status_code}"
                )

            result = poll_response.json()

            if result["status"] == "completed":
                return self._parse_assemblyai_response(result)
            elif result["status"] == "error":
                raise TranscriptionError(f"AssemblyAI error: {result.get('error')}")

            time.sleep(5)
            attempts += 1

        raise TranscriptionError("AssemblyAI transcription timed out")

    def _parse_assemblyai_response(self, response: dict) -> RawTranscript:
        """Parse AssemblyAI response into RawTranscript format."""
        segments: list[TranscriptSegment] = []

        for utterance in response.get("utterances", []):
            speaker_id = f"speaker_{utterance.get('speaker', 'unknown')}"

            words = [
                WordTiming(
                    text=w["text"],
                    start_time=w["start"] / 1000.0,
                    end_time=w["end"] / 1000.0,
                    confidence=w.get("confidence"),
                )
                for w in utterance.get("words", [])
            ]

            segment = TranscriptSegment(
                text=utterance.get("text", ""),
                start_time=utterance["start"] / 1000.0,
                end_time=utterance["end"] / 1000.0,
                speaker=TranscriptSpeaker(id=speaker_id),
                confidence=utterance.get("confidence"),
                words=words,
            )
            segments.append(segment)

        return RawTranscript(
            id="transcript:source:temp_assemblyai",
            source_id="",
            language_code=response.get("language_code", "en"),
            transcription_service="assemblyai",
            segments=segments,
            audio_duration_seconds=response.get("audio_duration"),
        )

    def _transcribe_deepgram(self, url: str, language: str = "en") -> RawTranscript:
        """Transcribe using Deepgram API with URL.

        Args:
            url: Presigned URL to the video file
            language: Language code

        Returns:
            RawTranscript with transcription results
        """
        api_url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": "nova-2",
            "language": language,
            "diarize": "true",
            "punctuate": "true",
            "utterances": "true",
        }

        response = self._client.post(
            api_url,
            params=params,
            headers={"Authorization": f"Token {self.api_key}"},
            json={"url": url},
        )

        if response.status_code != 200:
            raise TranscriptionError(
                f"Deepgram API error: {response.status_code} - {response.text}"
            )

        return self._parse_deepgram_response(response.json())

    def _parse_deepgram_response(self, response: dict) -> RawTranscript:
        """Parse Deepgram response into RawTranscript format."""
        segments: list[TranscriptSegment] = []

        results = response.get("results", {})
        for utterance in results.get("utterances", []):
            speaker_id = f"speaker_{utterance.get('speaker', 0)}"

            segment = TranscriptSegment(
                text=utterance.get("transcript", ""),
                start_time=utterance.get("start", 0.0),
                end_time=utterance.get("end", 0.0),
                speaker=TranscriptSpeaker(id=speaker_id),
                confidence=utterance.get("confidence"),
            )
            segments.append(segment)

        metadata = response.get("metadata", {})
        return RawTranscript(
            id="transcript:source:temp_deepgram",
            source_id="",
            language_code=metadata.get("language", "en"),
            transcription_service="deepgram",
            transcription_model=metadata.get("model_info", {}).get("name"),
            segments=segments,
            audio_duration_seconds=metadata.get("duration"),
        )

    def _merge_word_segments(
        self,
        segments: list[TranscriptSegment],
        pause_threshold: float = 1.0,
    ) -> list[TranscriptSegment]:
        """Merge word-level segments into utterance-level segments.

        Args:
            segments: Word-level segments to merge
            pause_threshold: Seconds of pause to start a new segment

        Returns:
            Merged utterance-level segments
        """
        if not segments:
            return []

        merged: list[TranscriptSegment] = []
        current_words: list[TranscriptSegment] = [segments[0]]
        current_speaker = segments[0].speaker.id if segments[0].speaker else None

        for segment in segments[1:]:
            speaker_id = segment.speaker.id if segment.speaker else None
            gap = segment.start_time - current_words[-1].end_time

            # Start new segment on speaker change or long pause
            if speaker_id != current_speaker or gap > pause_threshold:
                merged.append(self._combine_words(current_words))
                current_words = [segment]
                current_speaker = speaker_id
            else:
                current_words.append(segment)

        if current_words:
            merged.append(self._combine_words(current_words))

        return merged

    def _combine_words(self, words: list[TranscriptSegment]) -> TranscriptSegment:
        """Combine word segments into a single segment."""
        text = " ".join(w.text for w in words)

        confidences = [w.confidence for w in words if w.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return TranscriptSegment(
            text=text,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            speaker=words[0].speaker,
            confidence=avg_confidence,
            words=[
                WordTiming(
                    text=w.text,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    confidence=w.confidence,
                )
                for w in words
            ],
        )

    def _save_transcript(self, source: Source, transcript: RawTranscript) -> Path:
        """Save transcript to source directory.

        Args:
            source: The Source entity
            transcript: The transcription result

        Returns:
            Path to the saved transcript file
        """
        # Get source directory path
        # source:deploy:20250119_vinci_01/gopro_01
        deployment_dir = source.deployment_id.replace(":", "_")
        source_name = source.id.split("/")[-1]

        source_path = self.data_dir / deployment_dir / "sources" / source_name
        source_path.mkdir(parents=True, exist_ok=True)

        transcript_path = source_path / "raw_transcript.json"
        transcript_path.write_text(transcript.model_dump_json(indent=2))

        return transcript_path

    def update_source_status(
        self,
        source: Source,
        status: TranscriptStatus,
    ) -> None:
        """Update the transcript status of a source.

        Args:
            source: The Source entity to update
            status: The new transcript status
        """
        source.transcript_status = status

        # Save updated source
        deployment_dir = source.deployment_id.replace(":", "_")
        source_name = source.id.split("/")[-1]

        source_path = self.data_dir / deployment_dir / "sources" / source_name
        source_json = source_path / "source.json"

        if source_json.exists():
            source_json.write_text(source.model_dump_json(indent=2))

    def transcribe_and_update(
        self,
        source: Source,
        provider: str | None = None,
        language: str = "en",
    ) -> RawTranscript:
        """Transcribe a source and update its status.

        Convenience method that wraps transcribe() with status updates.

        Args:
            source: The Source entity to transcribe
            provider: Override the default provider
            language: Language code

        Returns:
            RawTranscript with transcription results

        Raises:
            TranscriptionError: If transcription fails
        """
        # Update status to processing
        self.update_source_status(source, TranscriptStatus.PROCESSING)

        try:
            transcript = self.transcribe(source, provider, language)
            self.update_source_status(source, TranscriptStatus.COMPLETE)
            return transcript
        except Exception as e:
            self.update_source_status(source, TranscriptStatus.FAILED)
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
