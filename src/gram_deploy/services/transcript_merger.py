"""Transcript Merger - combines raw transcripts into unified canonical transcript.

Responsible for:
- Converting timestamps to canonical time
- Detecting and removing duplicate utterances
- Merging adjacent segments from same speaker
- Creating CanonicalUtterance entities
"""

from typing import Optional
from difflib import SequenceMatcher

from gram_deploy.models import (
    CanonicalUtterance,
    CanonicalWord,
    RawTranscript,
    Source,
    SpeakerMapping,
    UtteranceSource,
)


class TranscriptMerger:
    """Merges raw transcripts into a unified canonical transcript."""

    def __init__(
        self,
        duplicate_overlap_threshold: float = 0.8,
        duplicate_text_threshold: float = 0.8,
        merge_gap_threshold_ms: int = 2000,
    ):
        """Initialize the merger.

        Args:
            duplicate_overlap_threshold: Min time overlap ratio to consider duplicate
            duplicate_text_threshold: Min text similarity to consider duplicate
            merge_gap_threshold_ms: Max gap between segments to merge (same speaker)
        """
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.duplicate_text_threshold = duplicate_text_threshold
        self.merge_gap_threshold_ms = merge_gap_threshold_ms

    def merge(
        self,
        sources: list[Source],
        transcripts: list[RawTranscript],
        speaker_mappings: list[SpeakerMapping],
    ) -> list[CanonicalUtterance]:
        """Create the unified canonical transcript.

        Args:
            sources: Sources with canonical offsets
            transcripts: Raw transcripts to merge
            speaker_mappings: Speaker resolution mappings

        Returns:
            List of CanonicalUtterance entities in chronological order
        """
        if not transcripts:
            return []

        deployment_id = sources[0].deployment_id
        source_map = {s.id: s for s in sources}
        mapping_lookup = self._build_mapping_lookup(speaker_mappings)

        # Convert all segments to canonical utterances
        all_utterances: list[CanonicalUtterance] = []

        for transcript in transcripts:
            source = source_map.get(transcript.source_id)
            if not source:
                continue

            offset_ms = source.canonical_offset_ms

            for i, segment in enumerate(transcript.segments):
                # Convert timestamps to canonical
                canonical_start = int(segment.start_time * 1000) + offset_ms
                canonical_end = int(segment.end_time * 1000) + offset_ms

                # Resolve speaker
                speaker_id = None
                speaker_confidence = 0.0
                if segment.speaker:
                    mapping_key = f"{transcript.source_id}/{segment.speaker.id}"
                    mapping = mapping_lookup.get(mapping_key)
                    if mapping:
                        speaker_id = mapping.resolved_person_id
                        speaker_confidence = mapping.confidence

                # Convert word timings
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

                utterance = CanonicalUtterance(
                    id=CanonicalUtterance.generate_id(deployment_id),
                    deployment_id=deployment_id,
                    text=segment.text,
                    canonical_start_ms=canonical_start,
                    canonical_end_ms=canonical_end,
                    speaker_id=speaker_id,
                    speaker_confidence=speaker_confidence,
                    sources=[
                        UtteranceSource(
                            source_id=transcript.source_id,
                            local_start_time=segment.start_time,
                            local_end_time=segment.end_time,
                            raw_segment_index=i,
                        )
                    ],
                    words=words,
                    is_duplicate=False,
                )
                all_utterances.append(utterance)

        # Sort by canonical start time
        all_utterances.sort(key=lambda u: u.canonical_start_ms)

        # Detect and handle duplicates
        merged_utterances = self._remove_duplicates(all_utterances)

        # Merge adjacent segments from same speaker
        final_utterances = self._merge_adjacent(merged_utterances)

        return final_utterances

    def _build_mapping_lookup(
        self,
        mappings: list[SpeakerMapping],
    ) -> dict[str, SpeakerMapping]:
        """Build lookup dict for speaker mappings."""
        lookup = {}
        for mapping in mappings:
            key = f"{mapping.source_id}/{mapping.raw_speaker_id}"
            lookup[key] = mapping
        return lookup

    def _remove_duplicates(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[CanonicalUtterance]:
        """Remove duplicate utterances captured by multiple sources.

        When duplicates are found, keep the one from the source with
        better audio quality and mark it as a duplicate.
        """
        if not utterances:
            return []

        result: list[CanonicalUtterance] = []
        skip_indices: set[int] = set()

        for i, utterance_a in enumerate(utterances):
            if i in skip_indices:
                continue

            # Find all utterances that might be duplicates
            duplicates = [utterance_a]
            duplicate_indices = [i]

            for j, utterance_b in enumerate(utterances[i + 1:], start=i + 1):
                if j in skip_indices:
                    continue

                if self._is_duplicate(utterance_a, utterance_b):
                    duplicates.append(utterance_b)
                    duplicate_indices.append(j)
                    skip_indices.add(j)

            if len(duplicates) > 1:
                # Merge duplicates - keep best, add sources from others
                best = self._select_best_utterance(duplicates)
                best.is_duplicate = True

                # Add sources from other duplicates
                for dup in duplicates:
                    if dup.id != best.id:
                        best.sources.extend(dup.sources)

                result.append(best)
            else:
                result.append(utterance_a)

        return result

    def _is_duplicate(
        self,
        a: CanonicalUtterance,
        b: CanonicalUtterance,
    ) -> bool:
        """Check if two utterances are duplicates."""
        # Must have same or similar speaker
        if a.speaker_id and b.speaker_id and a.speaker_id != b.speaker_id:
            return False

        # Must have sufficient time overlap
        overlap_start = max(a.canonical_start_ms, b.canonical_start_ms)
        overlap_end = min(a.canonical_end_ms, b.canonical_end_ms)
        overlap_duration = max(0, overlap_end - overlap_start)

        min_duration = min(a.duration_ms, b.duration_ms)
        if min_duration == 0:
            return False

        overlap_ratio = overlap_duration / min_duration
        if overlap_ratio < self.duplicate_overlap_threshold:
            return False

        # Must have similar text
        text_similarity = self._text_similarity(a.text, b.text)
        if text_similarity < self.duplicate_text_threshold:
            return False

        return True

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two text strings using SequenceMatcher."""
        return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    def _select_best_utterance(
        self,
        utterances: list[CanonicalUtterance],
    ) -> CanonicalUtterance:
        """Select the best utterance from a set of duplicates.

        Prefers utterances with:
        1. Higher speaker confidence
        2. More word-level timing data
        3. Longer text (more complete)
        """
        def score(u: CanonicalUtterance) -> tuple:
            word_count = len(u.words) if u.words else 0
            return (u.speaker_confidence, word_count, len(u.text))

        return max(utterances, key=score)

    def _merge_adjacent(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[CanonicalUtterance]:
        """Merge adjacent segments from the same speaker."""
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
        """Merge two adjacent utterances into one."""
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

    def resolve_overlapping_speech(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[CanonicalUtterance]:
        """Handle cases where multiple people speak simultaneously.

        Rather than interleaving, preserves both speakers' utterances
        with overlapping time ranges.
        """
        # Sort by start time
        sorted_utterances = sorted(utterances, key=lambda u: u.canonical_start_ms)

        # Mark overlapping speech
        for i, current in enumerate(sorted_utterances):
            for other in sorted_utterances[i + 1:]:
                if other.canonical_start_ms >= current.canonical_end_ms:
                    break  # No more overlaps possible

                # Check for overlap with different speaker
                if current.speaker_id != other.speaker_id:
                    # Both are overlapping speech - could annotate this
                    pass

        return sorted_utterances
