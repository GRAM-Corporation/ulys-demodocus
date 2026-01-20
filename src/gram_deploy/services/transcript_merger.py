"""Transcript Merger Service - creates canonical transcript from multiple sources.

Responsible for:
- Loading transcripts and speaker mappings from all sources
- Converting source-local times to canonical times
- Detecting duplicate utterances from overlapping sources
- Resolving conflicts by selecting best version
- Producing a unified, time-ordered canonical transcript
"""

import json
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

from gram_deploy.models import (
    CanonicalUtterance,
    CanonicalWord,
    Deployment,
    RawTranscript,
    Source,
    SpeakerMapping,
    TranscriptSegment,
    UtteranceSource,
)


class TranscriptMerger:
    """Merges transcripts from multiple sources into a canonical timeline."""

    def __init__(
        self,
        data_dir: str,
        duplicate_overlap_threshold: float = 0.5,
        duplicate_text_threshold: float = 0.85,
        merge_gap_threshold_ms: int = 2000,
    ):
        """Initialize the merger.

        Args:
            data_dir: Root data directory (e.g., ./deployments)
            duplicate_overlap_threshold: Min time overlap ratio to consider duplicate
            duplicate_text_threshold: Min text similarity to consider duplicate (0.85 per spec)
            merge_gap_threshold_ms: Max gap between segments to merge (same speaker)
        """
        self.data_dir = Path(data_dir)
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.duplicate_text_threshold = duplicate_text_threshold
        self.merge_gap_threshold_ms = merge_gap_threshold_ms

    def merge_transcripts(
        self,
        deployment: Deployment,
        sources: Optional[list[Source]] = None,
        transcripts: Optional[list[RawTranscript]] = None,
        speaker_mappings: Optional[dict[str, list[SpeakerMapping]]] = None,
    ) -> list[CanonicalUtterance]:
        """Merge all source transcripts into a canonical transcript.

        Args:
            deployment: The Deployment entity
            sources: Optional list of sources (loaded from disk if not provided)
            transcripts: Optional list of raw transcripts (loaded from disk if not provided)
            speaker_mappings: Optional mapping of source_id to list of SpeakerMapping
                              (loaded from disk if not provided)

        Returns:
            List of CanonicalUtterance sorted by canonical_start_ms
        """
        # Load data if not provided
        if sources is None:
            sources = self._load_sources(deployment)

        if not sources:
            return []

        if transcripts is None:
            transcripts = self._load_transcripts(deployment, sources)

        if speaker_mappings is None:
            speaker_mappings = self._load_speaker_mappings(deployment, sources)

        # Build lookup maps
        source_map = {s.id: s for s in sources}
        transcript_map = {t.source_id: t for t in transcripts}

        # Build flat speaker mapping lookup
        mapping_lookup = self._build_mapping_lookup(speaker_mappings)

        # Convert all segments to canonical utterances
        all_utterances: list[CanonicalUtterance] = []

        for source in sources:
            transcript = transcript_map.get(source.id)
            if not transcript:
                continue

            for idx, segment in enumerate(transcript.segments):
                utterance = self._convert_to_canonical_time(
                    segment=segment,
                    segment_index=idx,
                    source=source,
                    deployment=deployment,
                    mapping_lookup=mapping_lookup,
                )
                all_utterances.append(utterance)

        # Sort by canonical start time
        all_utterances.sort(key=lambda u: u.canonical_start_ms)

        # Detect and resolve duplicates
        if len(sources) > 1:
            duplicates_groups = self._detect_duplicates(all_utterances)
            all_utterances = self._resolve_conflicts(all_utterances, duplicates_groups)

        # Merge adjacent segments from same speaker
        all_utterances = self._merge_adjacent(all_utterances)

        return all_utterances

    def merge(
        self,
        sources: list[Source],
        transcripts: list[RawTranscript],
        speaker_mappings: list[SpeakerMapping],
    ) -> list[CanonicalUtterance]:
        """Create the unified canonical transcript (legacy method).

        Args:
            sources: Sources with canonical offsets
            transcripts: Raw transcripts to merge
            speaker_mappings: Speaker resolution mappings

        Returns:
            List of CanonicalUtterance entities in chronological order
        """
        if not transcripts or not sources:
            return []

        deployment_id = sources[0].deployment_id

        # Create a mock deployment for the new method
        deployment = Deployment(
            id=deployment_id,
            location="",
            date="2025-01-01",
            sources=[s.id for s in sources],
        )

        # Convert flat list to dict for new method
        mappings_dict: dict[str, list[SpeakerMapping]] = {}
        for mapping in speaker_mappings:
            if mapping.source_id not in mappings_dict:
                mappings_dict[mapping.source_id] = []
            mappings_dict[mapping.source_id].append(mapping)

        return self.merge_transcripts(
            deployment=deployment,
            sources=sources,
            transcripts=transcripts,
            speaker_mappings=mappings_dict,
        )

    def _convert_to_canonical_time(
        self,
        segment: TranscriptSegment,
        segment_index: int,
        source: Source,
        deployment: Deployment,
        mapping_lookup: dict[str, SpeakerMapping],
    ) -> CanonicalUtterance:
        """Convert a raw transcript segment to a canonical utterance.

        Applies source.canonical_offset_ms to convert local times to canonical.

        Args:
            segment: The raw transcript segment
            segment_index: Index of segment in transcript's segment list
            source: The source this segment came from
            deployment: The deployment entity
            mapping_lookup: Dictionary mapping "source_id/speaker_id" to SpeakerMapping

        Returns:
            CanonicalUtterance with canonical times and resolved speaker
        """
        offset_ms = source.canonical_offset_ms

        # Convert timestamps to canonical
        canonical_start_ms = int(segment.start_time * 1000) + offset_ms
        canonical_end_ms = int(segment.end_time * 1000) + offset_ms

        # Resolve speaker using mappings
        speaker_id, speaker_confidence = self._apply_speaker_mapping(
            segment, source.id, mapping_lookup
        )

        # Convert word timings if present
        words = None
        if segment.words:
            words = [
                CanonicalWord(
                    text=w.text,
                    canonical_start_ms=int(w.start_time * 1000) + offset_ms,
                    canonical_end_ms=int(w.end_time * 1000) + offset_ms,
                )
                for w in segment.words
            ]

        # Create utterance source reference
        utterance_source = UtteranceSource(
            source_id=source.id,
            local_start_time=segment.start_time,
            local_end_time=segment.end_time,
            raw_segment_index=segment_index,
        )

        return CanonicalUtterance(
            id=CanonicalUtterance.generate_id(deployment.id),
            deployment_id=deployment.id,
            text=segment.text,
            canonical_start_ms=canonical_start_ms,
            canonical_end_ms=canonical_end_ms,
            speaker_id=speaker_id,
            speaker_confidence=speaker_confidence,
            sources=[utterance_source],
            words=words,
            is_duplicate=False,
        )

    def _apply_speaker_mapping(
        self,
        segment: TranscriptSegment,
        source_id: str,
        mapping_lookup: dict[str, SpeakerMapping],
    ) -> tuple[Optional[str], float]:
        """Apply speaker mapping to resolve raw speaker ID.

        Args:
            segment: The transcript segment with potential speaker
            source_id: The source ID
            mapping_lookup: Dictionary mapping "source_id/speaker_id" to SpeakerMapping

        Returns:
            (speaker_id, confidence) - speaker_id is "unknown_N" if unmapped
        """
        if not segment.speaker:
            return None, 0.0

        raw_speaker_id = segment.speaker.id
        mapping_key = f"{source_id}/{raw_speaker_id}"
        mapping = mapping_lookup.get(mapping_key)

        if mapping and mapping.resolved_person_id:
            return mapping.resolved_person_id, mapping.confidence
        else:
            # Handle unmapped speakers as "unknown_N"
            return f"unknown_{raw_speaker_id}", 0.0

    def _build_mapping_lookup(
        self,
        mappings: dict[str, list[SpeakerMapping]] | list[SpeakerMapping],
    ) -> dict[str, SpeakerMapping]:
        """Build lookup dict for speaker mappings.

        Args:
            mappings: Either dict[source_id, list[SpeakerMapping]] or flat list

        Returns:
            Dictionary mapping "source_id/speaker_id" to SpeakerMapping
        """
        lookup: dict[str, SpeakerMapping] = {}

        if isinstance(mappings, dict):
            for source_id, source_mappings in mappings.items():
                for mapping in source_mappings:
                    key = f"{source_id}/{mapping.raw_speaker_id}"
                    lookup[key] = mapping
        else:
            # Handle flat list (legacy)
            for mapping in mappings:
                key = f"{mapping.source_id}/{mapping.raw_speaker_id}"
                lookup[key] = mapping

        return lookup

    def _detect_duplicates(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[list[int]]:
        """Find overlapping utterances from different sources.

        Same speaker + similar text + overlapping time = duplicate.

        Args:
            utterances: List of canonical utterances sorted by start time

        Returns:
            List of duplicate groups (each group is a list of indices)
        """
        if not utterances:
            return []

        duplicate_groups: list[list[int]] = []
        processed: set[int] = set()

        for i, utterance_a in enumerate(utterances):
            if i in processed:
                continue

            group = [i]

            for j, utterance_b in enumerate(utterances[i + 1:], start=i + 1):
                if j in processed:
                    continue

                # Check if outside time window (optimization)
                if utterance_b.canonical_start_ms > utterance_a.canonical_end_ms + 2000:
                    break

                if self._is_duplicate(utterance_a, utterance_b):
                    group.append(j)
                    processed.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(i)

        return duplicate_groups

    def _is_duplicate(
        self,
        a: CanonicalUtterance,
        b: CanonicalUtterance,
    ) -> bool:
        """Check if two utterances are duplicates.

        Criteria:
        - Same speaker (or both unknown, or one is unknown)
        - Overlapping time
        - Similar text (>= threshold)

        Args:
            a: First utterance
            b: Second utterance

        Returns:
            True if they are duplicates
        """
        # Check speaker compatibility
        speaker_match = self._speakers_compatible(a.speaker_id, b.speaker_id)
        if not speaker_match:
            return False

        # Check time overlap
        overlap_start = max(a.canonical_start_ms, b.canonical_start_ms)
        overlap_end = min(a.canonical_end_ms, b.canonical_end_ms)
        overlap_duration = max(0, overlap_end - overlap_start)

        min_duration = min(a.duration_ms, b.duration_ms)
        if min_duration == 0:
            return False

        overlap_ratio = overlap_duration / min_duration
        if overlap_ratio < self.duplicate_overlap_threshold:
            return False

        # Check text similarity
        text_similarity = self._text_similarity(a.text, b.text)
        if text_similarity < self.duplicate_text_threshold:
            return False

        return True

    def _speakers_compatible(
        self,
        speaker_a: Optional[str],
        speaker_b: Optional[str],
    ) -> bool:
        """Check if two speaker IDs are compatible for duplicate detection.

        Compatible means:
        - Both are the same resolved person
        - One or both are unknown (allows matching uncertain speakers)
        - Both are None
        """
        if speaker_a is None and speaker_b is None:
            return True

        if speaker_a is None or speaker_b is None:
            return True

        # If one is unknown, it could match anyone
        if speaker_a.startswith("unknown_") or speaker_b.startswith("unknown_"):
            return True

        return speaker_a == speaker_b

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two text strings.

        Uses SequenceMatcher for accurate text comparison.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not text_a or not text_b:
            return 0.0

        return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    def _resolve_conflicts(
        self,
        utterances: list[CanonicalUtterance],
        duplicate_groups: list[list[int]],
    ) -> list[CanonicalUtterance]:
        """Resolve conflicts between duplicate utterances.

        For each duplicate group, picks the best version based on:
        1. Higher transcription confidence (speaker confidence)
        2. Better audio quality source (more word timings)
        3. More complete text (longer)

        Marks others as alternates by consolidating their sources.

        Args:
            utterances: All utterances
            duplicate_groups: Groups of duplicate indices

        Returns:
            Deduplicated list of utterances
        """
        if not duplicate_groups:
            return utterances

        # Build set of indices to skip
        skip_indices: set[int] = set()
        replacements: dict[int, CanonicalUtterance] = {}

        for group in duplicate_groups:
            # Get all utterances in this group
            group_utterances = [(idx, utterances[idx]) for idx in group]

            # Select best utterance
            best_idx, best = self._select_best_utterance(group_utterances)

            # Collect all sources from duplicates
            all_sources = []
            for idx, utt in group_utterances:
                all_sources.extend(utt.sources)

            # Create merged utterance
            merged = CanonicalUtterance(
                id=best.id,
                deployment_id=best.deployment_id,
                text=best.text,
                canonical_start_ms=best.canonical_start_ms,
                canonical_end_ms=best.canonical_end_ms,
                speaker_id=best.speaker_id,
                speaker_confidence=best.speaker_confidence,
                sources=all_sources,
                words=best.words,
                is_duplicate=True,
            )

            # Mark all in group as skip, replace best index with merged
            for idx in group:
                skip_indices.add(idx)

            replacements[best_idx] = merged

        # Build result list
        result: list[CanonicalUtterance] = []
        for i, utterance in enumerate(utterances):
            if i in skip_indices:
                if i in replacements:
                    result.append(replacements[i])
            else:
                result.append(utterance)

        return result

    def _select_best_utterance(
        self,
        utterances: list[tuple[int, CanonicalUtterance]],
    ) -> tuple[int, CanonicalUtterance]:
        """Select the best utterance from a set of duplicates.

        Prefers utterances with:
        1. Higher speaker confidence
        2. More word-level timing data (indicates better transcription quality)
        3. Longer text (more complete)

        Args:
            utterances: List of (index, utterance) tuples

        Returns:
            (index, best_utterance) tuple
        """
        def score(item: tuple[int, CanonicalUtterance]) -> tuple:
            idx, u = item
            word_count = len(u.words) if u.words else 0
            return (u.speaker_confidence, word_count, len(u.text))

        return max(utterances, key=score)

    def _merge_adjacent(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[CanonicalUtterance]:
        """Merge adjacent segments from the same speaker.

        Args:
            utterances: List of utterances sorted by time

        Returns:
            List with adjacent same-speaker segments merged
        """
        if not utterances:
            return []

        result: list[CanonicalUtterance] = []
        current = utterances[0]

        for next_utterance in utterances[1:]:
            # Check if should merge
            gap_ms = next_utterance.canonical_start_ms - current.canonical_end_ms
            same_speaker = (
                current.speaker_id is not None
                and current.speaker_id == next_utterance.speaker_id
                and not current.speaker_id.startswith("unknown_")
            )

            if same_speaker and gap_ms <= self.merge_gap_threshold_ms:
                # Merge into current
                current = self._merge_utterances(current, next_utterance)
            else:
                result.append(current)
                current = next_utterance

        result.append(current)
        return result

    def _merge_utterances(
        self,
        a: CanonicalUtterance,
        b: CanonicalUtterance,
    ) -> CanonicalUtterance:
        """Merge two adjacent utterances into one.

        Args:
            a: First utterance
            b: Second utterance (comes after a)

        Returns:
            Merged utterance
        """
        merged_text = f"{a.text} {b.text}"

        merged_words = None
        if a.words or b.words:
            merged_words = (a.words or []) + (b.words or [])

        merged_sources = a.sources + b.sources

        return CanonicalUtterance(
            id=a.id,  # Keep first ID
            deployment_id=a.deployment_id,
            text=merged_text,
            canonical_start_ms=a.canonical_start_ms,
            canonical_end_ms=b.canonical_end_ms,
            speaker_id=a.speaker_id,
            speaker_confidence=max(a.speaker_confidence, b.speaker_confidence),
            sources=merged_sources,
            words=merged_words,
            is_duplicate=a.is_duplicate or b.is_duplicate,
        )

    def _load_sources(self, deployment: Deployment) -> list[Source]:
        """Load all sources for a deployment from disk.

        Args:
            deployment: The Deployment entity

        Returns:
            List of Source entities
        """
        sources: list[Source] = []
        deploy_dir = self._get_deployment_dir(deployment.id)

        for source_id in deployment.sources:
            # Parse source ID: source:deploy:20250119_vinci_01/gopro_01
            parts = source_id.replace("source:", "").split("/")
            if len(parts) != 2:
                continue

            device_part = parts[1]
            source_path = deploy_dir / "sources" / device_part / "source.json"

            if source_path.exists():
                data = json.loads(source_path.read_text())
                sources.append(Source.model_validate(data))

        return sources

    def _load_transcripts(
        self,
        deployment: Deployment,
        sources: list[Source],
    ) -> list[RawTranscript]:
        """Load raw transcripts for all sources.

        Args:
            deployment: The Deployment entity
            sources: List of Source entities

        Returns:
            List of RawTranscript entities
        """
        transcripts: list[RawTranscript] = []
        deploy_dir = self._get_deployment_dir(deployment.id)

        for source in sources:
            # Parse source ID to get device part
            parts = source.id.replace("source:", "").split("/")
            if len(parts) != 2:
                continue

            device_part = parts[1]

            # Try both possible transcript file names
            for filename in ["transcript.json", "raw_transcript.json"]:
                transcript_path = deploy_dir / "sources" / device_part / filename
                if transcript_path.exists():
                    data = json.loads(transcript_path.read_text())
                    transcripts.append(RawTranscript.model_validate(data))
                    break

        return transcripts

    def _load_speaker_mappings(
        self,
        deployment: Deployment,
        sources: list[Source],
    ) -> dict[str, list[SpeakerMapping]]:
        """Load speaker mappings for all sources.

        Args:
            deployment: The Deployment entity
            sources: List of Source entities

        Returns:
            Dictionary mapping source_id to list of SpeakerMappings
        """
        result: dict[str, list[SpeakerMapping]] = {}
        deploy_dir = self._get_deployment_dir(deployment.id)

        for source in sources:
            # Parse source ID to get device part
            parts = source.id.replace("source:", "").split("/")
            if len(parts) != 2:
                continue

            device_part = parts[1]
            mappings_path = deploy_dir / "sources" / device_part / "speaker_mappings.json"

            if mappings_path.exists():
                data = json.loads(mappings_path.read_text())
                mappings = []
                for item in data:
                    mapping = SpeakerMapping(
                        raw_speaker_id=item["raw_speaker_id"],
                        deployment_id=deployment.id,
                        source_id=source.id,
                        resolved_person_id=item.get("person_id"),
                        confidence=item.get("confidence", 0.0),
                        method=item.get("method", "unresolved"),
                        verified=False,
                    )
                    mappings.append(mapping)
                result[source.id] = mappings

        return result

    def _get_deployment_dir(self, deployment_id: str) -> Path:
        """Get the directory path for a deployment.

        Args:
            deployment_id: Deployment ID (e.g., deploy:20250119_vinci_01)

        Returns:
            Path to deployment directory
        """
        # Convert deploy:20250119_vinci_01 to deploy_20250119_vinci_01
        dir_name = deployment_id.replace(":", "_")
        return self.data_dir / dir_name

    def save_canonical_transcript(
        self,
        deployment: Deployment,
        utterances: list[CanonicalUtterance],
    ) -> Path:
        """Save the canonical transcript to the deployment directory.

        Writes to deployment/canonical_transcript.json

        Args:
            deployment: The Deployment entity
            utterances: List of canonical utterances

        Returns:
            Path to the saved file
        """
        deploy_dir = self._get_deployment_dir(deployment.id)
        output_path = deploy_dir / "canonical_transcript.json"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize utterances
        data = [u.model_dump() for u in utterances]
        output_path.write_text(json.dumps(data, indent=2, default=str))

        return output_path

    def load_canonical_transcript(
        self,
        deployment: Deployment,
    ) -> Optional[list[CanonicalUtterance]]:
        """Load existing canonical transcript if available.

        Args:
            deployment: The Deployment entity

        Returns:
            List of CanonicalUtterance or None if not found
        """
        deploy_dir = self._get_deployment_dir(deployment.id)
        transcript_path = deploy_dir / "canonical_transcript.json"

        if not transcript_path.exists():
            return None

        data = json.loads(transcript_path.read_text())
        return [CanonicalUtterance.model_validate(item) for item in data]

    def resolve_overlapping_speech(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[CanonicalUtterance]:
        """Handle cases where multiple people speak simultaneously.

        Rather than interleaving, preserves both speakers' utterances
        with overlapping time ranges.

        Args:
            utterances: List of utterances

        Returns:
            Same utterances, sorted by start time
        """
        return sorted(utterances, key=lambda u: u.canonical_start_ms)
