"""Speaker Resolution Service - maps abstract speaker IDs to known team members.

Responsible for:
- Extracting voice samples from transcripts
- Computing voice embeddings for speaker identification
- Matching speakers to known persons using voice similarity
- Using context clues (name mentions) for resolution
- Analyzing speaker patterns and frequency for identification
"""

import json
from pathlib import Path
from typing import Optional, Union
import re

from gram_deploy.models import (
    Deployment,
    Person,
    RawTranscript,
    ResolutionMethod,
    Source,
    SpeakerMapping,
    VoiceSample,
)


class SpeakerResolutionService:
    """Resolves speaker identities from transcription to known team members."""

    def __init__(self, people_registry_path: str, data_dir: Optional[str] = None):
        """Initialize the service.

        Args:
            people_registry_path: Path to the global people registry JSON file
            data_dir: Root directory for deployment data (inferred from registry_path if not provided)
        """
        self.registry_path = Path(people_registry_path)
        self.data_dir = Path(data_dir) if data_dir else self.registry_path.parent
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
        deployment: Union[Deployment, str],
        sources: Optional[list[Source]] = None,
        transcripts: Optional[list[RawTranscript]] = None,
    ) -> list[SpeakerMapping]:
        """Resolve speaker identities for a deployment.

        Can be called in two ways:
        1. resolve_speakers(deployment) - loads sources and transcripts internally
        2. resolve_speakers(deployment_id, sources, transcripts) - uses provided data

        Args:
            deployment: Deployment object or deployment ID string
            sources: Optional sources in the deployment (loaded if not provided)
            transcripts: Optional raw transcripts with speaker IDs (loaded if not provided)

        Returns:
            List of SpeakerMapping connecting raw IDs to Persons
        """
        # Handle both Deployment object and deployment_id string
        if isinstance(deployment, Deployment):
            deployment_id = deployment.id
            # Load sources and transcripts if not provided
            if sources is None:
                sources = self._load_sources(deployment)
            if transcripts is None:
                transcripts = self._load_transcripts(deployment)
        else:
            # Assume it's a deployment_id string (backward compatibility)
            deployment_id = deployment
            if sources is None:
                sources = []
            if transcripts is None:
                transcripts = []

        mappings: list[SpeakerMapping] = []
        transcript_map = {t.source_id: t for t in transcripts}

        # Collect all raw speaker IDs with their transcripts
        raw_speakers: dict[str, list[tuple[str, RawTranscript]]] = {}
        for transcript in transcripts:
            for speaker_id in transcript.speaker_ids:
                key = f"{transcript.source_id}/{speaker_id}"
                if key not in raw_speakers:
                    raw_speakers[key] = []
                raw_speakers[key].append((speaker_id, transcript))

        # Get team members from deployment for pattern matching
        team_members = []
        if isinstance(deployment, Deployment) and deployment.team_members:
            team_members = [self._people.get(pid) for pid in deployment.team_members]
            team_members = [p for p in team_members if p is not None]

        # Try to resolve each speaker
        for full_id, occurrences in raw_speakers.items():
            source_id = full_id.rsplit("/", 1)[0]
            raw_id = full_id.rsplit("/", 1)[1]
            transcript = occurrences[0][1]

            # Try voice matching first if we have embeddings
            person_id, confidence, method = self._resolve_by_voice(
                source_id, raw_id, transcript
            )

            # Try context inference
            if person_id is None:
                person_id, confidence, method = self._resolve_by_context(
                    raw_id, transcript
                )

            # Try pattern matching with team composition
            if person_id is None and team_members:
                person_id, confidence, method = self._match_by_pattern(
                    raw_speakers, list(self._people.values()), team_members
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

    def _load_sources(self, deployment: Deployment) -> list[Source]:
        """Load sources for a deployment from disk.

        Args:
            deployment: The deployment to load sources for

        Returns:
            List of Source entities
        """
        sources = []
        deploy_dir = self._get_deployment_dir(deployment.id)

        for source_id in deployment.sources:
            # Extract device part from source_id: source:deploy:20250119_vinci_01/gopro_01
            device_part = source_id.split("/")[-1]
            source_path = deploy_dir / "sources" / device_part / "source.json"

            if source_path.exists():
                data = json.loads(source_path.read_text())
                sources.append(Source.model_validate(data))

        return sources

    def _load_transcripts(self, deployment: Deployment) -> list[RawTranscript]:
        """Load raw transcripts for a deployment from disk.

        Args:
            deployment: The deployment to load transcripts for

        Returns:
            List of RawTranscript entities
        """
        transcripts = []
        deploy_dir = self._get_deployment_dir(deployment.id)

        for source_id in deployment.sources:
            device_part = source_id.split("/")[-1]
            transcript_path = deploy_dir / "sources" / device_part / "raw_transcript.json"

            if transcript_path.exists():
                data = json.loads(transcript_path.read_text())
                transcripts.append(RawTranscript.model_validate(data))

        return transcripts

    def _get_deployment_dir(self, deployment_id: str) -> Path:
        """Get the directory for a deployment."""
        dir_name = deployment_id.replace(":", "_")
        return self.data_dir / dir_name

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

    def _match_by_voice_embedding(
        self,
        speaker_audio: list[float],
        person: Person,
    ) -> float:
        """Compare speaker audio embedding against a person's stored voice embedding.

        Uses cosine similarity to compare voice embeddings.

        Args:
            speaker_audio: Voice embedding extracted from speaker segments
            person: Person to compare against

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not speaker_audio or not person.voice_embedding:
            return 0.0

        try:
            import numpy as np

            # Compute cosine similarity
            speaker_vec = np.array(speaker_audio)
            person_vec = np.array(person.voice_embedding)

            # Normalize vectors
            speaker_norm = np.linalg.norm(speaker_vec)
            person_norm = np.linalg.norm(person_vec)

            if speaker_norm == 0 or person_norm == 0:
                return 0.0

            similarity = np.dot(speaker_vec, person_vec) / (speaker_norm * person_norm)

            # Convert from [-1, 1] to [0, 1] range
            normalized_similarity = (similarity + 1) / 2

            return float(normalized_similarity)

        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _match_by_context(
        self,
        transcript: RawTranscript,
        person: Person,
    ) -> tuple[Optional[str], float]:
        """Look for context clues that identify a specific person in a transcript.

        Searches for:
        - Name mentions near speaker turns
        - Role-specific language ("as CTO...", "I'll handle the code...")
        - Person aliases mentioned

        Args:
            transcript: The raw transcript to search
            person: The person to look for

        Returns:
            Tuple of (raw_speaker_id if found, confidence score)
        """
        # Build list of names to search for this person
        names_to_find = [person.name.lower()]
        names_to_find.extend([a.lower() for a in person.aliases])
        # Add first name
        first_name = person.name.split()[0].lower()
        if first_name not in names_to_find:
            names_to_find.append(first_name)

        # Build role-specific patterns
        role_patterns = []
        if person.role:
            role_lower = person.role.lower()
            role_patterns = [
                rf"as (?:the )?{role_lower}",
                rf"i'm (?:the )?{role_lower}",
                rf"being (?:the )?{role_lower}",
            ]

        # Score each speaker based on context
        speaker_scores: dict[str, float] = {}

        for segment in transcript.segments:
            if not segment.speaker:
                continue

            speaker_id = segment.speaker.id
            text = segment.text.lower()

            if speaker_id not in speaker_scores:
                speaker_scores[speaker_id] = 0.0

            # Check for self-identification with person's name
            for name in names_to_find:
                patterns = [
                    rf"this is {name}",
                    rf"i'm {name}",
                    rf"i am {name}",
                    rf"{name} here",
                    rf"it's {name}",
                ]
                for pattern in patterns:
                    if re.search(pattern, text):
                        speaker_scores[speaker_id] += 0.8

            # Check for role-specific language
            for pattern in role_patterns:
                if re.search(pattern, text):
                    speaker_scores[speaker_id] += 0.3

        # Find the speaker with the highest score
        if speaker_scores:
            best_speaker = max(speaker_scores, key=speaker_scores.get)
            best_score = speaker_scores[best_speaker]
            if best_score > 0:
                return best_speaker, min(best_score, 1.0)

        return None, 0.0

    def _match_by_pattern(
        self,
        speakers: dict[str, list[tuple[str, RawTranscript]]],
        people: list[Person],
        team_members: Optional[list[Person]] = None,
    ) -> tuple[Optional[str], float, ResolutionMethod]:
        """Analyze speaker frequency and patterns to match against expected team composition.

        This method uses heuristics like:
        - Speaker frequency (most frequent speakers are likely key team members)
        - Number of sources a speaker appears in
        - Expected team size vs detected speaker count

        Args:
            speakers: Dictionary mapping speaker IDs to their occurrences
            people: List of all known people
            team_members: Optional list of expected team members for this deployment

        Returns:
            Tuple of (person_id if matched, confidence, method)
        """
        if not speakers or not people:
            return None, 0.0, ResolutionMethod.UNRESOLVED

        # Calculate speaking frequency for each raw speaker
        speaker_stats: dict[str, dict] = {}

        for full_id, occurrences in speakers.items():
            # Count total segments and unique sources
            total_segments = 0
            source_ids = set()

            for _, transcript in occurrences:
                source_ids.add(transcript.source_id)
                raw_id = full_id.rsplit("/", 1)[1]
                segment_count = sum(
                    1 for s in transcript.segments
                    if s.speaker and s.speaker.id == raw_id
                )
                total_segments += segment_count

            speaker_stats[full_id] = {
                "total_segments": total_segments,
                "source_count": len(source_ids),
                "sources": source_ids,
            }

        # If we have expected team members, try to match by frequency
        if team_members:
            # Sort speakers by frequency (most segments first)
            sorted_speakers = sorted(
                speaker_stats.items(),
                key=lambda x: x[1]["total_segments"],
                reverse=True
            )

            # For now, we can only suggest a match if there's a clear pattern
            # e.g., if we expect 2 team members and have 2 dominant speakers
            num_dominant_speakers = len([
                s for s in sorted_speakers
                if s[1]["total_segments"] > 5  # Arbitrary threshold
            ])

            if num_dominant_speakers == len(team_members) and len(team_members) == 1:
                # Single team member expected and single dominant speaker
                # This is a weak signal but can be used
                return team_members[0].id, 0.3, ResolutionMethod.CONTEXT_INFERENCE

        return None, 0.0, ResolutionMethod.UNRESOLVED

    def save_mappings(
        self,
        deployment_id: str,
        mappings: list[SpeakerMapping],
    ) -> None:
        """Save speaker mappings to source directories.

        Writes mappings to {source_path}/speaker_mappings.json for each source
        that has mappings.

        Args:
            deployment_id: The deployment ID
            mappings: List of speaker mappings to save
        """
        deploy_dir = self._get_deployment_dir(deployment_id)

        # Group mappings by source
        by_source: dict[str, list[SpeakerMapping]] = {}
        for mapping in mappings:
            if mapping.source_id not in by_source:
                by_source[mapping.source_id] = []
            by_source[mapping.source_id].append(mapping)

        # Save mappings for each source
        for source_id, source_mappings in by_source.items():
            # Extract device part from source_id
            device_part = source_id.split("/")[-1]
            source_dir = deploy_dir / "sources" / device_part

            if source_dir.exists():
                mappings_path = source_dir / "speaker_mappings.json"
                data = [
                    {
                        "raw_speaker_id": m.raw_speaker_id,
                        "person_id": m.resolved_person_id,
                        "confidence": m.confidence,
                        "method": m.method.value if hasattr(m.method, 'value') else m.method,
                    }
                    for m in source_mappings
                ]
                mappings_path.write_text(json.dumps(data, indent=2))

    def load_mappings(
        self,
        deployment_id: str,
        source_id: str,
    ) -> list[SpeakerMapping]:
        """Load speaker mappings from a source directory.

        Args:
            deployment_id: The deployment ID
            source_id: The source ID to load mappings for

        Returns:
            List of SpeakerMapping entities
        """
        deploy_dir = self._get_deployment_dir(deployment_id)
        device_part = source_id.split("/")[-1]
        mappings_path = deploy_dir / "sources" / device_part / "speaker_mappings.json"

        if not mappings_path.exists():
            return []

        data = json.loads(mappings_path.read_text())
        mappings = []

        for item in data:
            mapping = SpeakerMapping(
                raw_speaker_id=item["raw_speaker_id"],
                deployment_id=deployment_id,
                source_id=source_id,
                resolved_person_id=item.get("person_id"),
                confidence=item.get("confidence", 0.0),
                method=ResolutionMethod(item.get("method", "unresolved")),
                verified=False,
            )
            mappings.append(mapping)

        return mappings
