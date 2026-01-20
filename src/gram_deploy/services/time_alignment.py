"""Time Alignment Service - synchronizes multiple sources to a canonical timeline.

Responsible for:
- Computing time offsets between sources using audio fingerprints
- Fallback alignment using transcript matching
- Metadata-based alignment as last resort
- Verification of alignment quality
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import numpy as np

from gram_deploy.models import RawTranscript, Source, TimeAlignment


@dataclass
class AlignmentIssue:
    """An issue found during alignment verification."""

    source_id: str
    description: str
    severity: str  # warning, error
    suggested_fix: Optional[str] = None


class TimeAlignmentService:
    """Synchronizes multiple video sources to a canonical timeline."""

    def __init__(self, cache_dir: str):
        """Initialize the alignment service.

        Args:
            cache_dir: Directory for caching alignment computations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_alignment(
        self,
        sources: list[Source],
        transcripts: list[RawTranscript],
        audio_fingerprints: Optional[dict[str, bytes]] = None,
    ) -> TimeAlignment:
        """Compute time alignment for all sources.

        Args:
            sources: List of Source entities
            transcripts: Corresponding RawTranscript entities
            audio_fingerprints: Optional pre-computed fingerprints {source_id: fingerprint}

        Returns:
            TimeAlignment with offsets for each source
        """
        if not sources:
            raise ValueError("At least one source is required")

        # Find earliest recording start as canonical zero
        canonical_start = self._find_earliest_start(sources)

        # Initialize alignment
        alignment = TimeAlignment(
            deployment_id=sources[0].deployment_id,
            canonical_start_time=canonical_start,
            source_offsets={},
            confidence_scores={},
            alignment_methods={},
            cross_correlations=[],
        )

        # Single source: trivial alignment
        if len(sources) == 1:
            source = sources[0]
            alignment.source_offsets[source.id] = 0
            alignment.confidence_scores[source.id] = 1.0
            alignment.alignment_methods[source.id] = "single_source"
            return alignment

        # Try audio fingerprint alignment first
        if audio_fingerprints:
            self._align_by_audio(sources, audio_fingerprints, alignment)

        # Use transcript matching for sources not yet aligned
        unaligned = [s for s in sources if s.id not in alignment.source_offsets]
        if unaligned and transcripts:
            transcript_map = {t.source_id: t for t in transcripts}
            self._align_by_transcript(unaligned, transcript_map, alignment)

        # Fallback to metadata for remaining sources
        still_unaligned = [s for s in sources if s.id not in alignment.source_offsets]
        if still_unaligned:
            self._align_by_metadata(still_unaligned, canonical_start, alignment)

        # Compute canonical end time
        alignment.canonical_end_time = self._compute_end_time(sources, alignment)

        return alignment

    def apply_alignment(self, alignment: TimeAlignment, sources: list[Source]) -> None:
        """Apply computed alignment to sources.

        Args:
            alignment: The computed TimeAlignment
            sources: Sources to update with alignment data
        """
        for source in sources:
            if source.id in alignment.source_offsets:
                source.canonical_offset_ms = alignment.source_offsets[source.id]
                source.alignment_confidence = alignment.confidence_scores.get(source.id, 0.0)
                source.alignment_method = alignment.alignment_methods.get(source.id, "unknown")

    def verify_alignment(
        self,
        alignment: TimeAlignment,
        transcripts: list[RawTranscript],
        tolerance_ms: int = 2000,
    ) -> list[AlignmentIssue]:
        """Verify alignment quality by checking transcript consistency.

        Args:
            alignment: The TimeAlignment to verify
            transcripts: Transcripts to check for consistency
            tolerance_ms: Maximum acceptable time difference for matching text

        Returns:
            List of alignment issues found
        """
        issues: list[AlignmentIssue] = []

        # Find matching text across transcripts
        matches = self._find_transcript_matches(transcripts)

        for match in matches:
            source_a, source_b, text, time_a, time_b = match

            # Convert to canonical time
            canonical_a = time_a * 1000 + alignment.get_offset(source_a)
            canonical_b = time_b * 1000 + alignment.get_offset(source_b)

            diff = abs(canonical_a - canonical_b)
            if diff > tolerance_ms:
                issues.append(AlignmentIssue(
                    source_id=source_b,
                    description=f"Text '{text[:50]}...' differs by {diff}ms from {source_a}",
                    severity="warning" if diff < tolerance_ms * 2 else "error",
                    suggested_fix=f"Adjust offset by {canonical_a - canonical_b}ms",
                ))

        return issues

    def _find_earliest_start(self, sources: list[Source]) -> datetime:
        """Find the earliest recording start time across sources."""
        # Try to extract from file metadata
        earliest = datetime.utcnow()

        for source in sources:
            if source.files:
                # Use file creation time as approximation
                for f in source.files:
                    path = Path(f.file_path)
                    if path.exists():
                        mtime = datetime.fromtimestamp(path.stat().st_mtime)
                        # Subtract duration to get start time
                        start = mtime.timestamp() - f.duration_seconds
                        file_start = datetime.fromtimestamp(start)
                        if file_start < earliest:
                            earliest = file_start

        return earliest

    def _compute_end_time(
        self,
        sources: list[Source],
        alignment: TimeAlignment,
    ) -> datetime:
        """Compute the canonical end time from aligned sources."""
        max_end_ms = 0

        for source in sources:
            offset = alignment.get_offset(source.id)
            source_end_ms = int(source.total_duration * 1000) + offset
            max_end_ms = max(max_end_ms, source_end_ms)

        return datetime.fromtimestamp(
            alignment.canonical_start_time.timestamp() + max_end_ms / 1000
        )

    def _align_by_audio(
        self,
        sources: list[Source],
        fingerprints: dict[str, bytes],
        alignment: TimeAlignment,
    ) -> None:
        """Align sources using audio fingerprint cross-correlation."""
        # Use first source as reference
        ref_source = sources[0]
        ref_fp = fingerprints.get(ref_source.id)

        if not ref_fp:
            return

        alignment.source_offsets[ref_source.id] = 0
        alignment.confidence_scores[ref_source.id] = 1.0
        alignment.alignment_methods[ref_source.id] = "audio_fingerprint"

        for source in sources[1:]:
            source_fp = fingerprints.get(source.id)
            if not source_fp:
                continue

            # Cross-correlate fingerprints
            offset, confidence = self._cross_correlate_fingerprints(ref_fp, source_fp)

            if confidence >= 0.5:
                alignment.source_offsets[source.id] = offset
                alignment.confidence_scores[source.id] = confidence
                alignment.alignment_methods[source.id] = "audio_fingerprint"
                alignment.cross_correlations.append({
                    "source_a": ref_source.id,
                    "source_b": source.id,
                    "offset_ms": offset,
                    "confidence": confidence,
                })

    def _cross_correlate_fingerprints(
        self,
        fp_a: bytes,
        fp_b: bytes,
    ) -> tuple[int, float]:
        """Cross-correlate two audio fingerprints.

        Returns:
            (offset_ms, confidence) where offset is fp_b - fp_a
        """
        try:
            # Convert fingerprints to numpy arrays
            arr_a = np.frombuffer(fp_a, dtype=np.int32)
            arr_b = np.frombuffer(fp_b, dtype=np.int32)

            if len(arr_a) == 0 or len(arr_b) == 0:
                return 0, 0.0

            # Compute cross-correlation
            correlation = np.correlate(arr_a, arr_b, mode='full')
            max_idx = np.argmax(np.abs(correlation))

            # Convert index to time offset (assuming ~100ms per fingerprint unit)
            offset_units = max_idx - len(arr_b) + 1
            offset_ms = offset_units * 100

            # Confidence based on correlation peak
            max_corr = np.abs(correlation[max_idx])
            auto_corr = np.sqrt(np.sum(arr_a**2) * np.sum(arr_b**2))
            confidence = max_corr / auto_corr if auto_corr > 0 else 0.0

            return offset_ms, min(confidence, 1.0)

        except Exception:
            return 0, 0.0

    def _align_by_transcript(
        self,
        sources: list[Source],
        transcripts: dict[str, RawTranscript],
        alignment: TimeAlignment,
    ) -> None:
        """Align sources using matching transcript text."""
        # Find matching phrases across transcripts
        for source in sources:
            transcript = transcripts.get(source.id)
            if not transcript:
                continue

            # Look for matching text in already-aligned sources
            for aligned_id, offset in alignment.source_offsets.items():
                aligned_transcript = transcripts.get(aligned_id)
                if not aligned_transcript:
                    continue

                match_offset, confidence = self._find_text_offset(
                    transcript, aligned_transcript, offset
                )

                if confidence >= 0.4:
                    alignment.source_offsets[source.id] = match_offset
                    alignment.confidence_scores[source.id] = confidence
                    alignment.alignment_methods[source.id] = "transcript_match"
                    break

    def _find_text_offset(
        self,
        transcript_a: RawTranscript,
        transcript_b: RawTranscript,
        b_offset: int,
    ) -> tuple[int, float]:
        """Find time offset between transcripts by matching text.

        Returns:
            (offset_ms for a, confidence)
        """
        # Extract significant phrases (5+ words)
        phrases_a = [
            (s.text, s.start_time)
            for s in transcript_a.segments
            if len(s.text.split()) >= 5
        ]
        phrases_b = [
            (s.text, s.start_time)
            for s in transcript_b.segments
            if len(s.text.split()) >= 5
        ]

        matches: list[tuple[float, float]] = []

        for text_a, time_a in phrases_a:
            for text_b, time_b in phrases_b:
                similarity = self._text_similarity(text_a, text_b)
                if similarity > 0.8:
                    # Compute what offset would make these align
                    canonical_b = time_b * 1000 + b_offset
                    offset_a = canonical_b - time_a * 1000
                    matches.append((offset_a, similarity))

        if not matches:
            return 0, 0.0

        # Use median offset from matches
        offsets = [m[0] for m in matches]
        median_offset = int(np.median(offsets))
        confidence = len(matches) / max(len(phrases_a), 1) * 0.5 + 0.5

        return median_offset, min(confidence, 1.0)

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two text strings."""
        # Simple word overlap similarity
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _align_by_metadata(
        self,
        sources: list[Source],
        canonical_start: datetime,
        alignment: TimeAlignment,
    ) -> None:
        """Fallback alignment using file metadata."""
        for source in sources:
            if source.files:
                # Use first file's modification time
                path = Path(source.files[0].file_path)
                if path.exists():
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    # Estimate start time
                    start = mtime.timestamp() - source.files[0].duration_seconds
                    offset_ms = int((start - canonical_start.timestamp()) * 1000)

                    alignment.source_offsets[source.id] = offset_ms
                    alignment.confidence_scores[source.id] = 0.3
                    alignment.alignment_methods[source.id] = "metadata"
                else:
                    # No metadata available, assume zero offset
                    alignment.source_offsets[source.id] = 0
                    alignment.confidence_scores[source.id] = 0.1
                    alignment.alignment_methods[source.id] = "unaligned"

    def _find_transcript_matches(
        self,
        transcripts: list[RawTranscript],
    ) -> list[tuple[str, str, str, float, float]]:
        """Find matching text across transcripts.

        Returns:
            List of (source_a, source_b, text, time_a, time_b) tuples
        """
        matches = []

        for i, transcript_a in enumerate(transcripts):
            for transcript_b in transcripts[i + 1:]:
                for seg_a in transcript_a.segments:
                    for seg_b in transcript_b.segments:
                        if self._text_similarity(seg_a.text, seg_b.text) > 0.8:
                            matches.append((
                                transcript_a.source_id,
                                transcript_b.source_id,
                                seg_a.text,
                                seg_a.start_time,
                                seg_b.start_time,
                            ))

        return matches

    def save_alignment(self, alignment: TimeAlignment, path: str) -> None:
        """Save alignment to disk."""
        Path(path).write_text(alignment.model_dump_json(indent=2))

    def load_alignment(self, path: str) -> TimeAlignment:
        """Load alignment from disk."""
        data = json.loads(Path(path).read_text())
        return TimeAlignment.model_validate(data)
