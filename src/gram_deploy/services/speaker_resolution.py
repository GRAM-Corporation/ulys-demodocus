"""Speaker Resolution Service - maps abstract speaker IDs to known team members.

Responsible for:
- Extracting voice samples from transcripts
- Computing voice embeddings for speaker identification
- Matching speakers to known persons using voice similarity
- Using context clues (name mentions) for resolution
- Analyzing speaker patterns and frequency
"""

import json
from pathlib import Path
from typing import Optional
import re
from collections import Counter

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

    def _match_by_pattern(
        self,
        speakers: list[str],
        transcripts: list[RawTranscript],
        team_members: list[str],
    ) -> dict[str, tuple[Optional[str], float, ResolutionMethod]]:
        """Match speakers to people using frequency and pattern analysis.

        Analyzes:
        - Speaker frequency (total speaking time)
        - Speaking patterns (who speaks after whom)
        - Expected team composition from deployment.team_members

        Args:
            speakers: List of raw speaker IDs from transcripts
            transcripts: Raw transcripts with segment data
            team_members: List of person IDs expected in the deployment

        Returns:
            Dictionary mapping raw_speaker_id to (person_id, confidence, method)
        """
        results: dict[str, tuple[Optional[str], float, ResolutionMethod]] = {}

        # Calculate speaking time per speaker across all transcripts
        speaker_time: Counter = Counter()
        speaker_segments: dict[str, int] = {}

        for transcript in transcripts:
            for segment in transcript.segments:
                if segment.speaker:
                    speaker_id = segment.speaker.id
                    duration = segment.end_time - segment.start_time
                    speaker_time[speaker_id] += duration
                    speaker_segments[speaker_id] = speaker_segments.get(speaker_id, 0) + 1

        # Get expected people
        expected_people = [self._people.get(pid) for pid in team_members]
        expected_people = [p for p in expected_people if p is not None]

        if not expected_people or not speaker_time:
            return results

        # Sort speakers by speaking time (descending)
        ranked_speakers = speaker_time.most_common()

        # Match by role hints in the people data
        for speaker_id, _ in ranked_speakers:
            for person in expected_people:
                # Check if person is already matched
                if any(r[0] == person.id for r in results.values()):
                    continue

                # Match based on role-specific language patterns
                confidence = self._calculate_role_match_confidence(
                    speaker_id, person, transcripts
                )
                if confidence > 0.4:
                    results[speaker_id] = (person.id, confidence, ResolutionMethod.CONTEXT_INFERENCE)
                    break

        # If we have equal number of speakers and team members, try position matching
        if len(ranked_speakers) == len(expected_people):
            unmatched_speakers = [s for s, _ in ranked_speakers if s not in results]
            unmatched_people = [p for p in expected_people
                               if p.id not in [r[0] for r in results.values()]]

            # Tentative matching by speaking frequency (most talkative -> first in team list)
            for speaker_id, person in zip(unmatched_speakers, unmatched_people):
                results[speaker_id] = (person.id, 0.3, ResolutionMethod.CONTEXT_INFERENCE)

        return results

    def _calculate_role_match_confidence(
        self,
        speaker_id: str,
        person: Person,
        transcripts: list[RawTranscript],
    ) -> float:
        """Calculate confidence that a speaker matches a person based on role language.

        Args:
            speaker_id: The raw speaker ID
            person: The person to match against
            transcripts: Transcripts containing the speaker's segments

        Returns:
            Confidence score from 0.0 to 1.0
        """
        if not person.role:
            return 0.0

        # Get all text from this speaker
        speaker_text = ""
        for transcript in transcripts:
            for segment in transcript.segments:
                if segment.speaker and segment.speaker.id == speaker_id:
                    speaker_text += " " + segment.text

        speaker_text_lower = speaker_text.lower()

        # Define role-specific patterns
        role_patterns: dict[str, list[str]] = {
            "cto": [
                r"technical",
                r"architect",
                r"infrastructure",
                r"code",
                r"deploy",
                r"system",
                r"engineer",
                r"api",
                r"database",
            ],
            "ceo": [
                r"strategy",
                r"business",
                r"investor",
                r"board",
                r"company",
                r"growth",
                r"revenue",
            ],
            "coo": [
                r"operations",
                r"process",
                r"team",
                r"logistics",
                r"schedule",
            ],
            "engineer": [
                r"code",
                r"debug",
                r"bug",
                r"fix",
                r"implement",
                r"test",
            ],
        }

        role_lower = person.role.lower()
        patterns = role_patterns.get(role_lower, [])

        if not patterns:
            return 0.0

        # Count pattern matches
        matches = sum(1 for p in patterns if re.search(p, speaker_text_lower))
        confidence = min(matches / len(patterns), 1.0) * 0.6

        # Check for explicit role mentions
        if re.search(rf"as (the )?{role_lower}", speaker_text_lower):
            confidence += 0.3
        if re.search(rf"i('m| am) (the )?{role_lower}", speaker_text_lower):
            confidence += 0.4

        return min(confidence, 1.0)

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

    def save_mappings(
        self,
        mappings: list[SpeakerMapping],
        deployment: Deployment,
        base_path: str = "deployments",
    ) -> dict[str, Path]:
        """Save speaker mappings to their respective source directories.

        Writes mappings to {source_path}/speaker_mappings.json for each source.

        Args:
            mappings: List of speaker mappings to save
            deployment: The deployment these mappings belong to
            base_path: Base directory for deployments

        Returns:
            Dictionary mapping source_id to the path where mappings were saved
        """
        saved_paths: dict[str, Path] = {}

        # Group mappings by source
        by_source: dict[str, list[SpeakerMapping]] = {}
        for mapping in mappings:
            if mapping.source_id not in by_source:
                by_source[mapping.source_id] = []
            by_source[mapping.source_id].append(mapping)

        # Save each source's mappings
        for source_id, source_mappings in by_source.items():
            source_path = deployment.get_source_path(source_id, base_path)
            mappings_file = source_path / "speaker_mappings.json"

            # Create directory if needed
            source_path.mkdir(parents=True, exist_ok=True)

            # Serialize mappings
            data = {
                "deployment_id": deployment.id,
                "source_id": source_id,
                "mappings": [
                    {
                        "raw_speaker_id": m.raw_speaker_id,
                        "person_id": m.resolved_person_id,
                        "confidence": m.confidence,
                        "method": m.method.value if hasattr(m.method, 'value') else str(m.method),
                        "verified": m.verified,
                        "evidence_notes": m.evidence_notes,
                    }
                    for m in source_mappings
                ],
            }

            mappings_file.write_text(json.dumps(data, indent=2))
            saved_paths[source_id] = mappings_file

        return saved_paths

    def load_mappings(
        self,
        source_id: str,
        deployment: Deployment,
        base_path: str = "deployments",
    ) -> list[SpeakerMapping]:
        """Load speaker mappings for a source from disk.

        Args:
            source_id: The source ID to load mappings for
            deployment: The deployment the source belongs to
            base_path: Base directory for deployments

        Returns:
            List of SpeakerMapping objects
        """
        source_path = deployment.get_source_path(source_id, base_path)
        mappings_file = source_path / "speaker_mappings.json"

        if not mappings_file.exists():
            return []

        data = json.loads(mappings_file.read_text())
        mappings = []

        for m in data.get("mappings", []):
            mapping = SpeakerMapping(
                raw_speaker_id=m["raw_speaker_id"],
                deployment_id=deployment.id,
                source_id=source_id,
                resolved_person_id=m.get("person_id"),
                confidence=m.get("confidence", 0.0),
                method=ResolutionMethod(m.get("method", "unresolved")),
                verified=m.get("verified", False),
                evidence_notes=m.get("evidence_notes"),
            )
            mappings.append(mapping)

        return mappings

    def _match_by_voice_embedding(
        self,
        speaker_audio_path: str,
        speaker_start: float,
        speaker_end: float,
        person: Person,
    ) -> tuple[float, str]:
        """Compare a speaker's audio segment against a person's voice embedding.

        Uses resemblyzer or speechbrain for speaker recognition.

        Args:
            speaker_audio_path: Path to the audio file
            speaker_start: Start time in seconds
            speaker_end: End time in seconds
            person: Person to compare against

        Returns:
            Tuple of (similarity_score, method_description)
        """
        if person.voice_embedding is None:
            return 0.0, "no_embedding"

        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import numpy as np

            # Load and preprocess audio
            wav = preprocess_wav(speaker_audio_path)

            # Trim to segment
            sample_rate = 16000  # resemblyzer default
            start_sample = int(speaker_start * sample_rate)
            end_sample = int(speaker_end * sample_rate)
            wav_segment = wav[start_sample:end_sample]

            # Compute embedding for speaker segment
            encoder = VoiceEncoder()
            speaker_embedding = encoder.embed_utterance(wav_segment)

            # Compare with person's embedding using cosine similarity
            person_embedding = np.array(person.voice_embedding)
            similarity = np.dot(speaker_embedding, person_embedding) / (
                np.linalg.norm(speaker_embedding) * np.linalg.norm(person_embedding)
            )

            return float(similarity), "voice_embedding"

        except ImportError:
            return 0.0, "resemblyzer_not_available"
        except Exception as e:
            return 0.0, f"error: {str(e)}"

    def resolve_speakers_for_deployment(
        self,
        deployment: Deployment,
        transcripts: list[RawTranscript],
        sources: Optional[list[Source]] = None,
        base_path: str = "deployments",
        save: bool = True,
    ) -> list[SpeakerMapping]:
        """Resolve speaker identities for a deployment.

        This is the main entry point that accepts a Deployment object
        and handles loading transcripts and saving mappings.

        Args:
            deployment: The Deployment object
            transcripts: Raw transcripts with speaker IDs
            sources: Optional list of sources (for additional context)
            base_path: Base directory for deployments
            save: Whether to save mappings to disk

        Returns:
            List of SpeakerMapping connecting raw IDs to Persons
        """
        # Use the existing resolve_speakers method
        mappings = self.resolve_speakers(
            deployment_id=deployment.id,
            sources=sources or [],
            transcripts=transcripts,
        )

        # Apply pattern matching as a fallback for unresolved speakers
        unresolved = [m for m in mappings if not m.is_resolved]
        if unresolved and deployment.team_members:
            all_speakers = list({m.raw_speaker_id for m in mappings})
            pattern_results = self._match_by_pattern(
                speakers=all_speakers,
                transcripts=transcripts,
                team_members=deployment.team_members,
            )

            # Apply pattern results to unresolved mappings
            for mapping in unresolved:
                if mapping.raw_speaker_id in pattern_results:
                    person_id, confidence, method = pattern_results[mapping.raw_speaker_id]
                    if person_id:
                        mapping.resolved_person_id = person_id
                        mapping.confidence = confidence
                        mapping.method = method
                        mapping.evidence_notes = "Resolved via pattern matching"

        # Save mappings if requested
        if save:
            self.save_mappings(mappings, deployment, base_path)

        return mappings
