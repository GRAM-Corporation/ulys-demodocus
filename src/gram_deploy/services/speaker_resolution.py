"""Speaker Resolution Service - maps abstract speaker IDs to known team members.

Responsible for:
- Extracting voice samples from transcripts
- Computing voice embeddings for speaker identification
- Matching speakers to known persons using voice similarity
- Using context clues (name mentions) for resolution
"""

import json
from pathlib import Path
from typing import Optional
import re

from gram_deploy.models import (
    Person,
    RawTranscript,
    ResolutionMethod,
    Source,
    SpeakerMapping,
    VoiceSample,
)


class SpeakerResolutionService:
    """Resolves speaker identities from transcription to known team members."""

    def __init__(self, people_registry_path: str):
        """Initialize the service.

        Args:
            people_registry_path: Path to the global people registry JSON file
        """
        self.registry_path = Path(people_registry_path)
        self._people: dict[str, Person] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the people registry from disk."""
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            for person_data in data.get("people", []):
                person = Person.model_validate(person_data)
                self._people[person.id] = person

    def _save_registry(self) -> None:
        """Save the people registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "people": [p.model_dump() for p in self._people.values()]
        }
        self.registry_path.write_text(json.dumps(data, indent=2, default=str))

    def get_person(self, person_id: str) -> Optional[Person]:
        """Get a person by ID."""
        return self._people.get(person_id)

    def add_person(self, person: Person) -> None:
        """Add a person to the registry."""
        self._people[person.id] = person
        self._save_registry()

    def list_people(self) -> list[Person]:
        """List all people in the registry."""
        return list(self._people.values())

    def resolve_speakers(
        self,
        deployment_id: str,
        sources: list[Source],
        transcripts: list[RawTranscript],
    ) -> list[SpeakerMapping]:
        """Resolve speaker identities for a deployment.

        Args:
            deployment_id: The deployment ID
            sources: Sources in the deployment
            transcripts: Raw transcripts with speaker IDs

        Returns:
            List of SpeakerMapping connecting raw IDs to Persons
        """
        mappings: list[SpeakerMapping] = []
        transcript_map = {t.source_id: t for t in transcripts}

        # Collect all raw speaker IDs
        raw_speakers: dict[str, list[tuple[str, RawTranscript]]] = {}
        for transcript in transcripts:
            for speaker_id in transcript.speaker_ids:
                key = f"{transcript.source_id}/{speaker_id}"
                if key not in raw_speakers:
                    raw_speakers[key] = []
                raw_speakers[key].append((speaker_id, transcript))

        # Try to resolve each speaker
        for full_id, occurrences in raw_speakers.items():
            source_id = full_id.rsplit("/", 1)[0]
            raw_id = full_id.rsplit("/", 1)[1]
            transcript = occurrences[0][1]

            # Try voice matching first if we have embeddings
            person_id, confidence, method = self._resolve_by_voice(
                source_id, raw_id, transcript
            )

            # Fall back to context inference
            if person_id is None:
                person_id, confidence, method = self._resolve_by_context(
                    raw_id, transcript
                )

            mapping = SpeakerMapping(
                raw_speaker_id=raw_id,
                deployment_id=deployment_id,
                source_id=source_id,
                resolved_person_id=person_id,
                confidence=confidence,
                method=method,
                verified=False,
            )
            mappings.append(mapping)

        # Cross-reference across sources for consistency
        self._cross_reference_mappings(mappings, transcripts)

        return mappings

    def _resolve_by_voice(
        self,
        source_id: str,
        raw_speaker_id: str,
        transcript: RawTranscript,
    ) -> tuple[Optional[str], float, ResolutionMethod]:
        """Attempt to resolve speaker using voice matching.

        Returns:
            (person_id or None, confidence, method)
        """
        # Check if any people have voice embeddings
        people_with_embeddings = [
            p for p in self._people.values()
            if p.voice_embedding is not None
        ]

        if not people_with_embeddings:
            return None, 0.0, ResolutionMethod.UNRESOLVED

        # Would compute voice embedding and compare here
        # For now, return unresolved (implementation requires audio processing)
        return None, 0.0, ResolutionMethod.UNRESOLVED

    def _resolve_by_context(
        self,
        raw_speaker_id: str,
        transcript: RawTranscript,
    ) -> tuple[Optional[str], float, ResolutionMethod]:
        """Attempt to resolve speaker using context clues.

        Looks for patterns like:
        - "Hey Damion, can you..." (addressing)
        - "This is Damion, I'm..." (self-identification)
        - "Damion said..." followed by speaker change

        Returns:
            (person_id or None, confidence, method)
        """
        # Get segments for this speaker
        speaker_segments = [
            s for s in transcript.segments
            if s.speaker and s.speaker.id == raw_speaker_id
        ]

        if not speaker_segments:
            return None, 0.0, ResolutionMethod.UNRESOLVED

        # Build list of all known names
        known_names: dict[str, str] = {}
        for person in self._people.values():
            known_names[person.name.lower()] = person.id
            for alias in person.aliases:
                known_names[alias.lower()] = person.id
            # Add first name
            first_name = person.name.split()[0].lower()
            known_names[first_name] = person.id

        # Check for self-identification
        for segment in speaker_segments[:5]:  # Check first few utterances
            text = segment.text.lower()

            # Pattern: "This is [name]" or "I'm [name]" or "[name] here"
            for name, person_id in known_names.items():
                patterns = [
                    rf"this is {name}",
                    rf"i'm {name}",
                    rf"i am {name}",
                    rf"{name} here",
                    rf"it's {name}",
                ]
                for pattern in patterns:
                    if re.search(pattern, text):
                        return person_id, 0.8, ResolutionMethod.CONTEXT_INFERENCE

        # Check if speaker is addressed by name in preceding segment
        all_segments = sorted(transcript.segments, key=lambda s: s.start_time)
        for i, segment in enumerate(all_segments):
            if segment.speaker and segment.speaker.id == raw_speaker_id:
                # Check previous segment for addressing
                if i > 0:
                    prev = all_segments[i - 1]
                    prev_text = prev.text.lower()
                    for name, person_id in known_names.items():
                        # Pattern: "Hey [name]" or "[name], can you"
                        patterns = [
                            rf"hey {name}",
                            rf"{name},\s*can you",
                            rf"{name},\s*could you",
                            rf"ok {name}",
                            rf"okay {name}",
                        ]
                        for pattern in patterns:
                            if re.search(pattern, prev_text):
                                return person_id, 0.6, ResolutionMethod.CONTEXT_INFERENCE

        return None, 0.0, ResolutionMethod.UNRESOLVED

    def _cross_reference_mappings(
        self,
        mappings: list[SpeakerMapping],
        transcripts: list[RawTranscript],
    ) -> None:
        """Cross-reference mappings across sources for consistency.

        If same speaker is resolved to same person in multiple sources,
        increase confidence. If different, flag for review.
        """
        # Group by resolved person
        by_person: dict[str, list[SpeakerMapping]] = {}
        for mapping in mappings:
            if mapping.resolved_person_id:
                if mapping.resolved_person_id not in by_person:
                    by_person[mapping.resolved_person_id] = []
                by_person[mapping.resolved_person_id].append(mapping)

        # Boost confidence for consistent mappings
        for person_id, person_mappings in by_person.items():
            if len(person_mappings) > 1:
                # Same person identified in multiple sources
                for mapping in person_mappings:
                    mapping.confidence = min(mapping.confidence + 0.1, 1.0)

    def add_voice_sample(
        self,
        person_id: str,
        source_id: str,
        start_time: float,
        end_time: float,
        verified: bool = True,
    ) -> None:
        """Add a verified voice sample for a person.

        Args:
            person_id: The person ID
            source_id: Source ID where the sample is from
            start_time: Start time in source-local seconds
            end_time: End time in source-local seconds
            verified: Whether this sample is human-verified
        """
        person = self._people.get(person_id)
        if not person:
            raise ValueError(f"Person not found: {person_id}")

        sample = VoiceSample(
            source_id=source_id,
            start_time=start_time,
            end_time=end_time,
            verified=verified,
        )
        person.voice_samples.append(sample)
        self._save_registry()

    def compute_voice_embedding(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
    ) -> list[float]:
        """Compute a voice embedding for an audio segment.

        Uses resemblyzer or speechbrain for speaker recognition.

        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Voice embedding as list of floats
        """
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import numpy as np

            # Load and preprocess audio
            wav = preprocess_wav(audio_path)

            # Trim to segment
            sample_rate = 16000  # resemblyzer default
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            wav_segment = wav[start_sample:end_sample]

            # Compute embedding
            encoder = VoiceEncoder()
            embedding = encoder.embed_utterance(wav_segment)

            return embedding.tolist()

        except ImportError:
            # Resemblyzer not available
            return []
        except Exception:
            return []

    def update_person_embedding(self, person_id: str, audio_path: str) -> None:
        """Update a person's voice embedding from their samples.

        Args:
            person_id: The person ID
            audio_path: Base path to audio files
        """
        person = self._people.get(person_id)
        if not person or not person.voice_samples:
            return

        embeddings = []
        for sample in person.voice_samples:
            if sample.verified:
                embedding = self.compute_voice_embedding(
                    audio_path,
                    sample.start_time,
                    sample.end_time,
                )
                if embedding:
                    embeddings.append(embedding)

        if embeddings:
            # Average embeddings
            import numpy as np
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            person.voice_embedding = avg_embedding
            self._save_registry()
