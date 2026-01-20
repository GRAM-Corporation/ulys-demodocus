"""Transcription Service - speech-to-text with speaker diarization.

Responsible for:
- Sending audio to external transcription APIs (ElevenLabs, AssemblyAI, Whisper)
- Parsing responses into RawTranscript format
- Caching results to avoid redundant API calls
- Handling rate limiting and retries
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from gram_deploy.models import RawTranscript, Source, TranscriptSegment, TranscriptSpeaker, WordTiming


class TranscriptionService:
    """Handles speech-to-text transcription with speaker diarization."""

    PROVIDERS = ("elevenlabs", "assemblyai", "whisper", "deepgram")

    def __init__(self, provider: str, api_key: str, cache_dir: str):
        """Initialize the transcription service.

        Args:
            provider: Transcription provider (elevenlabs, assemblyai, whisper, deepgram)
            api_key: API key for the provider
            cache_dir: Directory for caching transcription results
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Must be one of {self.PROVIDERS}")

        self.provider = provider
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = httpx.Client(timeout=300.0)

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        enable_diarization: bool = True,
    ) -> RawTranscript:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Language code (default: en)
            enable_diarization: Whether to enable speaker diarization

        Returns:
            RawTranscript with segments and speaker identification
        """
        # Check cache first
        cache_key = self._compute_cache_key(audio_path, language, enable_diarization)
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        # Call provider-specific transcription
        if self.provider == "elevenlabs":
            result = self._transcribe_elevenlabs(audio_path, language, enable_diarization)
        elif self.provider == "assemblyai":
            result = self._transcribe_assemblyai(audio_path, language, enable_diarization)
        elif self.provider == "deepgram":
            result = self._transcribe_deepgram(audio_path, language, enable_diarization)
        else:
            raise NotImplementedError(f"Provider {self.provider} not yet implemented")

        # Cache result
        self._save_to_cache(cache_key, result)

        return result

    def transcribe_source(
        self,
        source: Source,
        audio_paths: list[str],
        language: str = "en",
    ) -> RawTranscript:
        """Transcribe all audio files for a source and merge.

        Args:
            source: The Source entity
            audio_paths: Paths to extracted audio files (same order as source.files)
            language: Language code

        Returns:
            Merged RawTranscript with adjusted timestamps
        """
        all_segments: list[TranscriptSegment] = []
        current_offset = 0.0

        for i, (audio_path, source_file) in enumerate(zip(audio_paths, source.files)):
            transcript = self.transcribe(audio_path, language)

            # Adjust timestamps for file position
            for segment in transcript.segments:
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
                    ] if segment.words else None,
                )
                all_segments.append(adjusted_segment)

            current_offset += source_file.duration_seconds

        return RawTranscript(
            id=RawTranscript.generate_id(source.id),
            source_id=source.id,
            language_code=language,
            transcription_service=self.provider,
            segments=all_segments,
            audio_duration_seconds=current_offset,
        )

    def _transcribe_elevenlabs(
        self,
        audio_path: str,
        language: str,
        enable_diarization: bool,
    ) -> RawTranscript:
        """Transcribe using ElevenLabs Scribe API."""
        url = "https://api.elevenlabs.io/v1/speech-to-text"

        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f, "audio/wav")}
            data = {
                "model_id": "scribe_v1",
                "language_code": language,
                "diarize": str(enable_diarization).lower(),
                "timestamps_granularity": "word",
            }

            response = self._client.post(
                url,
                headers={"xi-api-key": self.api_key},
                files=files,
                data=data,
            )

        if response.status_code == 429:
            # Rate limited - wait and retry
            time.sleep(60)
            return self._transcribe_elevenlabs(audio_path, language, enable_diarization)

        response.raise_for_status()
        result = response.json()

        return self._parse_elevenlabs_response(result, audio_path)

    def _parse_elevenlabs_response(self, response: dict, audio_path: str) -> RawTranscript:
        """Parse ElevenLabs API response into RawTranscript format."""
        segments: list[TranscriptSegment] = []

        for word_data in response.get("words", []):
            # ElevenLabs returns word-level data with speaker info
            # Group words into segments by speaker changes or pauses

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
            id=f"transcript:temp:{hashlib.md5(audio_path.encode()).hexdigest()[:8]}",
            source_id="",  # Will be set by caller
            language_code=response.get("language_code", "en"),
            transcription_service="elevenlabs",
            transcription_model="scribe_v1",
            segments=merged_segments,
            audio_duration_seconds=response.get("audio_duration"),
        )

    def _transcribe_assemblyai(
        self,
        audio_path: str,
        language: str,
        enable_diarization: bool,
    ) -> RawTranscript:
        """Transcribe using AssemblyAI API."""
        # Upload file
        upload_url = "https://api.assemblyai.com/v2/upload"
        with open(audio_path, "rb") as f:
            upload_response = self._client.post(
                upload_url,
                headers={"authorization": self.api_key},
                content=f.read(),
            )
        upload_response.raise_for_status()
        audio_url = upload_response.json()["upload_url"]

        # Request transcription
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        transcript_request = {
            "audio_url": audio_url,
            "language_code": language,
            "speaker_labels": enable_diarization,
        }

        response = self._client.post(
            transcript_url,
            headers={"authorization": self.api_key},
            json=transcript_request,
        )
        response.raise_for_status()
        transcript_id = response.json()["id"]

        # Poll for completion
        polling_url = f"{transcript_url}/{transcript_id}"
        while True:
            poll_response = self._client.get(
                polling_url,
                headers={"authorization": self.api_key},
            )
            poll_response.raise_for_status()
            result = poll_response.json()

            if result["status"] == "completed":
                return self._parse_assemblyai_response(result, audio_path)
            elif result["status"] == "error":
                raise RuntimeError(f"AssemblyAI error: {result.get('error')}")

            time.sleep(5)

    def _parse_assemblyai_response(self, response: dict, audio_path: str) -> RawTranscript:
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
            id=f"transcript:temp:{hashlib.md5(audio_path.encode()).hexdigest()[:8]}",
            source_id="",
            language_code=response.get("language_code", "en"),
            transcription_service="assemblyai",
            segments=segments,
            audio_duration_seconds=response.get("audio_duration"),
        )

    def _transcribe_deepgram(
        self,
        audio_path: str,
        language: str,
        enable_diarization: bool,
    ) -> RawTranscript:
        """Transcribe using Deepgram API."""
        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": "nova-2",
            "language": language,
            "diarize": str(enable_diarization).lower(),
            "punctuate": "true",
            "utterances": "true",
        }

        with open(audio_path, "rb") as f:
            response = self._client.post(
                url,
                params=params,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                content=f.read(),
            )

        response.raise_for_status()
        return self._parse_deepgram_response(response.json(), audio_path)

    def _parse_deepgram_response(self, response: dict, audio_path: str) -> RawTranscript:
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
            id=f"transcript:temp:{hashlib.md5(audio_path.encode()).hexdigest()[:8]}",
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
        """Merge word-level segments into utterance-level segments."""
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
        return TranscriptSegment(
            text=text,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            speaker=words[0].speaker,
            confidence=sum(w.confidence or 0 for w in words) / len(words) if words else None,
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

    def _compute_cache_key(
        self,
        audio_path: str,
        language: str,
        enable_diarization: bool,
    ) -> str:
        """Compute cache key for transcription results."""
        path = Path(audio_path)
        content_hash = hashlib.md5()

        # Hash file content (first/last chunks for speed)
        with open(path, "rb") as f:
            content_hash.update(f.read(8192))
            f.seek(-8192, 2)
            content_hash.update(f.read(8192))

        content_hash.update(language.encode())
        content_hash.update(str(enable_diarization).encode())
        content_hash.update(self.provider.encode())

        return content_hash.hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[RawTranscript]:
        """Load cached transcription result."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            return RawTranscript.model_validate(data)
        return None

    def _save_to_cache(self, cache_key: str, transcript: RawTranscript) -> None:
        """Save transcription result to cache."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        cache_path.write_text(transcript.model_dump_json(indent=2))
