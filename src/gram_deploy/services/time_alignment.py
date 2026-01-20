"""Time Alignment Service - synchronizes multiple sources to a canonical timeline.

Responsible for:
- Computing time offsets between sources using transcript matching
- Fallback alignment using file metadata
- Verification of alignment quality
"""

import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from gram_deploy.models import Deployment, RawTranscript, Source


class AlignmentResult(BaseModel):
    """Result of aligning a single source to the canonical timeline."""

    source_id: str
    canonical_offset_ms: int = Field(
        ..., description="Milliseconds to add to source-local time to get canonical time"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Alignment confidence")
    method: str = Field(
        ..., description="Alignment method: transcript, metadata, or single_source"
    )
    match_count: int = Field(
        default=0, ge=0, description="Number of matching phrases found (for transcript method)"
    )


@dataclass
class Match:
    """A matching phrase between two transcripts."""

    offset_ms: int
    similarity: float


@dataclass
class AlignmentIssue:
    """An issue found during alignment verification."""

    source_id: str
    description: str
    severity: str  # warning, error
    suggested_fix: Optional[str] = None


class TimeAlignmentService:
    """Synchronizes multiple video sources to a canonical timeline.

    Primary method is transcript-based alignment using overlapping speech
    patterns to compute time offsets between sources.
    """

    def __init__(self, data_dir: str):
        """Initialize the alignment service.

        Args:
            data_dir: Root data directory (e.g., ./deployments)
        """
        self.data_dir = Path(data_dir)

    def align_sources(
        self,
        deployment: Deployment,
        sources: Optional[list[Source]] = None,
        transcripts: Optional[list[RawTranscript]] = None,
    ) -> dict[str, AlignmentResult]:
        """Compute time alignment for all sources in a deployment.

        Args:
            deployment: The Deployment entity
            sources: Optional list of Source entities (loaded from disk if not provided)
            transcripts: Optional list of RawTranscript entities (loaded from disk if not provided)

        Returns:
            Dictionary mapping source_id to AlignmentResult
        """
        # Load sources if not provided
        if sources is None:
            sources = self._load_sources(deployment)

        if not sources:
            return {}

        # Handle single source case
        if len(sources) == 1:
            source = sources[0]
            return {
                source.id: AlignmentResult(
                    source_id=source.id,
                    canonical_offset_ms=0,
                    confidence=1.0,
                    method="single_source",
                    match_count=0,
                )
            }

        # Load transcripts if not provided
        if transcripts is None:
            transcripts = self._load_transcripts(deployment, sources)

        # Build transcript map
        transcript_map = {t.source_id: t for t in transcripts}

        # Use first source as reference
        reference_source = sources[0]
        results: dict[str, AlignmentResult] = {
            reference_source.id: AlignmentResult(
                source_id=reference_source.id,
                canonical_offset_ms=0,
                confidence=1.0,
                method="transcript",
                match_count=0,
            )
        }

        # Align other sources relative to reference
        reference_transcript = transcript_map.get(reference_source.id)

        for source in sources[1:]:
            source_transcript = transcript_map.get(source.id)

            # Try transcript alignment first
            if reference_transcript and source_transcript:
                offset_ms, confidence, match_count = self._align_by_transcript(
                    reference_transcript, source_transcript
                )

                if confidence >= 0.5:  # Minimum threshold for transcript alignment
                    results[source.id] = AlignmentResult(
                        source_id=source.id,
                        canonical_offset_ms=offset_ms,
                        confidence=confidence,
                        method="transcript",
                        match_count=match_count,
                    )
                    continue

            # Fall back to metadata alignment
            offset_ms, confidence = self._align_by_metadata(source, reference_source)
            results[source.id] = AlignmentResult(
                source_id=source.id,
                canonical_offset_ms=offset_ms,
                confidence=confidence,
                method="metadata",
                match_count=0,
            )

        return results

    def _align_by_transcript(
        self,
        transcript_a: RawTranscript,
        transcript_b: RawTranscript,
    ) -> tuple[int, float, int]:
        """Align two sources using transcript text matching.

        Uses fuzzy matching to find overlapping phrases and compute
        the time offset between transcripts.

        Args:
            transcript_a: Reference transcript (offset = 0)
            transcript_b: Transcript to align

        Returns:
            (offset_ms, confidence, match_count) where offset is what to add to
            transcript_b times to get canonical time
        """
        # Find matching segments
        matches = self._find_matches(transcript_a, transcript_b)

        if not matches:
            return 0, 0.0, 0

        # Single match - lower confidence
        if len(matches) == 1:
            return matches[0].offset_ms, self._confidence_for_matches(1), 1

        # Multiple matches - cluster offsets to find consensus
        offset_ms, confidence = self._calculate_offset_from_matches(matches)
        return offset_ms, confidence, len(matches)

    def _find_matches(
        self,
        transcript_a: RawTranscript,
        transcript_b: RawTranscript,
        similarity_threshold: float = 0.85,
    ) -> list[Match]:
        """Find matching phrases between two transcripts.

        Args:
            transcript_a: First transcript (reference)
            transcript_b: Second transcript
            similarity_threshold: Minimum similarity for a match (default 0.85)

        Returns:
            List of Match objects with offset and similarity
        """
        matches: list[Match] = []

        for seg_a in transcript_a.segments:
            # Skip very short segments
            if len(seg_a.text.split()) < 3:
                continue

            for seg_b in transcript_b.segments:
                if len(seg_b.text.split()) < 3:
                    continue

                similarity = self._fuzzy_ratio(seg_a.text, seg_b.text)

                if similarity > similarity_threshold:
                    # Offset: what to add to B's time to match A's time
                    # If seg_a is at 1000ms and seg_b is at 500ms,
                    # then offset = 1000 - 500 = 500ms (add 500ms to B)
                    offset = int(seg_a.start_time * 1000) - int(seg_b.start_time * 1000)
                    matches.append(Match(offset_ms=offset, similarity=similarity))

        return matches

    def _fuzzy_ratio(self, text_a: str, text_b: str) -> float:
        """Compute fuzzy similarity ratio between two strings.

        Uses a combination of word overlap (Jaccard) and character-level
        longest common subsequence ratio.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        # Normalize texts
        a = text_a.lower().strip()
        b = text_b.lower().strip()

        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        # Word-level Jaccard similarity
        words_a = set(a.split())
        words_b = set(b.split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = intersection / union if union > 0 else 0.0

        # Character-level ratio using LCS
        lcs_length = self._lcs_length(a, b)
        lcs_ratio = (2.0 * lcs_length) / (len(a) + len(b))

        # Combine both metrics (weight word overlap more heavily)
        return 0.6 * jaccard + 0.4 * lcs_ratio

    def _lcs_length(self, a: str, b: str) -> int:
        """Compute length of longest common subsequence.

        Uses space-optimized dynamic programming.

        Args:
            a: First string
            b: Second string

        Returns:
            Length of LCS
        """
        if not a or not b:
            return 0

        # Space optimization: only keep two rows
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def _calculate_offset_from_matches(
        self,
        matches: list[Match],
    ) -> tuple[int, float]:
        """Calculate consensus offset from multiple matches using clustering.

        Groups offsets into clusters and returns the median of the dominant
        cluster along with a confidence score.

        Args:
            matches: List of Match objects

        Returns:
            (offset_ms, confidence)
        """
        if not matches:
            return 0, 0.0

        if len(matches) == 1:
            return matches[0].offset_ms, self._confidence_for_matches(1)

        offsets = [m.offset_ms for m in matches]

        # Cluster offsets using a simple approach: group by proximity
        clusters = self._cluster_offsets(offsets, tolerance_ms=2000)

        if not clusters:
            return 0, 0.0

        # Find the largest cluster
        largest_cluster = max(clusters, key=len)

        # Use median of largest cluster
        median_offset = int(statistics.median(largest_cluster))

        # Confidence based on cluster tightness and size
        confidence = self._compute_cluster_confidence(largest_cluster, len(matches))

        return median_offset, confidence

    def _cluster_offsets(
        self,
        offsets: list[int],
        tolerance_ms: int = 2000,
    ) -> list[list[int]]:
        """Cluster offsets by proximity.

        Args:
            offsets: List of offset values in milliseconds
            tolerance_ms: Maximum difference to be in same cluster

        Returns:
            List of clusters, each cluster is a list of offsets
        """
        if not offsets:
            return []

        sorted_offsets = sorted(offsets)
        clusters: list[list[int]] = [[sorted_offsets[0]]]

        for offset in sorted_offsets[1:]:
            # Check if offset belongs to current cluster
            if abs(offset - clusters[-1][-1]) <= tolerance_ms:
                clusters[-1].append(offset)
            else:
                # Start new cluster
                clusters.append([offset])

        return clusters

    def _compute_cluster_confidence(
        self,
        cluster: list[int],
        total_matches: int,
    ) -> float:
        """Compute confidence score for a cluster.

        Args:
            cluster: List of offsets in the cluster
            total_matches: Total number of matches

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not cluster:
            return 0.0

        cluster_size = len(cluster)

        # Base confidence from match count
        base_confidence = self._confidence_for_matches(cluster_size)

        # Adjust for cluster tightness (lower std = higher confidence)
        if cluster_size > 1:
            try:
                std_dev = statistics.stdev(cluster)
                # Tightness bonus: lower std dev -> higher bonus
                # 0ms std -> +0.05, 1000ms std -> +0, 2000ms+ -> -0.05
                tightness_bonus = max(-0.05, min(0.05, 0.05 - std_dev / 20000))
                base_confidence = min(0.95, base_confidence + tightness_bonus)
            except statistics.StatisticsError:
                pass

        # Adjust for proportion of matches in cluster
        proportion = cluster_size / total_matches if total_matches > 0 else 0
        if proportion < 0.5:
            # Less than half matches agree - lower confidence
            base_confidence *= 0.8

        return round(base_confidence, 2)

    def _confidence_for_matches(self, match_count: int) -> float:
        """Get base confidence level for a given match count.

        Per spec:
        - 5+ matches: 0.85-0.95
        - 2-4 matches: 0.7-0.85
        - 1 match: 0.5-0.7

        Args:
            match_count: Number of matching phrases

        Returns:
            Base confidence score
        """
        if match_count >= 5:
            # Scale from 0.85 to 0.95 based on count (5->0.85, 10+->0.95)
            return min(0.95, 0.85 + (match_count - 5) * 0.02)
        elif match_count >= 2:
            # Scale from 0.7 to 0.85 based on count (2->0.7, 4->0.85)
            return 0.7 + (match_count - 2) * 0.05
        elif match_count == 1:
            return 0.6  # Middle of 0.5-0.7 range
        else:
            return 0.0

    def _align_by_metadata(
        self,
        source: Source,
        reference_source: Optional[Source] = None,
    ) -> tuple[int, float]:
        """Fallback alignment using file metadata.

        Parses video file creation timestamps to calculate offset.

        Args:
            source: Source to align
            reference_source: Reference source (optional, for relative offset)

        Returns:
            (offset_ms, confidence) with confidence in 0.3-0.5 range
        """
        source_start = self._get_source_start_time(source)
        ref_start = self._get_source_start_time(reference_source) if reference_source else None

        if source_start is None:
            # No metadata available
            return 0, 0.3

        if ref_start is None:
            # Have source start but no reference
            return 0, 0.4

        # Calculate offset relative to reference
        offset_ms = int((ref_start - source_start) * 1000)

        # Confidence based on whether we got actual timestamps
        confidence = 0.5 if source_start and ref_start else 0.3

        return offset_ms, confidence

    def _get_source_start_time(self, source: Optional[Source]) -> Optional[float]:
        """Get estimated start time for a source from file metadata.

        Args:
            source: Source entity

        Returns:
            Start time as Unix timestamp, or None if not available
        """
        if not source or not source.files:
            return None

        # Use first file's modification time minus its duration
        first_file = source.files[0]
        path = Path(first_file.file_path)

        if not path.exists():
            return None

        try:
            mtime = path.stat().st_mtime
            # Subtract duration to estimate start time
            return mtime - first_file.duration_seconds
        except OSError:
            return None

    def calculate_canonical_timeline(
        self,
        deployment: Deployment,
        alignments: dict[str, AlignmentResult],
    ) -> None:
        """Apply alignment results to sources and update deployment timeline.

        Sets canonical_offset_ms on each source and updates the deployment's
        canonical_start_time and canonical_end_time.

        Args:
            deployment: Deployment entity to update
            alignments: Alignment results from align_sources()
        """
        # Load sources
        sources = self._load_sources(deployment)

        if not sources:
            return

        # Find earliest and latest times across all sources
        canonical_start: Optional[datetime] = None
        canonical_end_ms: int = 0

        for source in sources:
            alignment = alignments.get(source.id)
            if not alignment:
                continue

            # Update source with alignment
            source.canonical_offset_ms = alignment.canonical_offset_ms
            source.alignment_confidence = alignment.confidence
            source.alignment_method = alignment.method

            # Calculate source's end in canonical time
            source_duration_ms = int(source.total_duration * 1000)
            source_end_canonical = source_duration_ms + alignment.canonical_offset_ms
            canonical_end_ms = max(canonical_end_ms, source_end_canonical)

            # Save updated source
            self._save_source(deployment, source)

        # Get canonical start from reference source metadata
        if sources and sources[0].files:
            start_time = self._get_source_start_time(sources[0])
            if start_time:
                canonical_start = datetime.fromtimestamp(start_time)

        # Update deployment
        if canonical_start:
            deployment.canonical_start_time = canonical_start
            deployment.canonical_end_time = datetime.fromtimestamp(
                canonical_start.timestamp() + canonical_end_ms / 1000
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
            # Parse source ID to get device part
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
            transcript_path = deploy_dir / "sources" / device_part / "transcript.json"

            if transcript_path.exists():
                data = json.loads(transcript_path.read_text())
                transcripts.append(RawTranscript.model_validate(data))

        return transcripts

    def _save_source(self, deployment: Deployment, source: Source) -> None:
        """Save a source to disk.

        Args:
            deployment: The Deployment entity
            source: The Source entity to save
        """
        # Parse source ID to get device part
        parts = source.id.replace("source:", "").split("/")
        if len(parts) != 2:
            return

        device_part = parts[1]
        deploy_dir = self._get_deployment_dir(deployment.id)
        source_path = deploy_dir / "sources" / device_part / "source.json"

        source_path.write_text(source.model_dump_json(indent=2))

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

    def verify_alignment(
        self,
        alignments: dict[str, AlignmentResult],
        transcripts: list[RawTranscript],
        tolerance_ms: int = 2000,
    ) -> list[AlignmentIssue]:
        """Verify alignment quality by checking transcript consistency.

        Args:
            alignments: Alignment results
            transcripts: Transcripts to check
            tolerance_ms: Maximum acceptable time difference

        Returns:
            List of alignment issues found
        """
        issues: list[AlignmentIssue] = []

        # Find matching text across transcripts and verify times align
        for i, transcript_a in enumerate(transcripts):
            for transcript_b in transcripts[i + 1:]:
                alignment_a = alignments.get(transcript_a.source_id)
                alignment_b = alignments.get(transcript_b.source_id)

                if not alignment_a or not alignment_b:
                    continue

                # Check for matching text
                for seg_a in transcript_a.segments:
                    for seg_b in transcript_b.segments:
                        similarity = self._fuzzy_ratio(seg_a.text, seg_b.text)

                        if similarity > 0.85:
                            # Convert to canonical time
                            canonical_a = (
                                int(seg_a.start_time * 1000)
                                + alignment_a.canonical_offset_ms
                            )
                            canonical_b = (
                                int(seg_b.start_time * 1000)
                                + alignment_b.canonical_offset_ms
                            )

                            diff = abs(canonical_a - canonical_b)
                            if diff > tolerance_ms:
                                issues.append(
                                    AlignmentIssue(
                                        source_id=transcript_b.source_id,
                                        description=(
                                            f"Text '{seg_a.text[:50]}...' differs "
                                            f"by {diff}ms from {transcript_a.source_id}"
                                        ),
                                        severity="warning" if diff < tolerance_ms * 2 else "error",
                                        suggested_fix=f"Adjust offset by {canonical_a - canonical_b}ms",
                                    )
                                )

        return issues

    def save_alignment_results(
        self,
        deployment: Deployment,
        alignments: dict[str, AlignmentResult],
    ) -> None:
        """Save alignment results to the deployment's cache directory.

        Args:
            deployment: The Deployment entity
            alignments: Alignment results to save
        """
        deploy_dir = self._get_deployment_dir(deployment.id)
        alignment_path = deploy_dir / "cache" / "alignment" / "alignment.json"
        alignment_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            source_id: result.model_dump()
            for source_id, result in alignments.items()
        }

        alignment_path.write_text(json.dumps(data, indent=2))

    def load_alignment_results(
        self,
        deployment: Deployment,
    ) -> Optional[dict[str, AlignmentResult]]:
        """Load cached alignment results.

        Args:
            deployment: The Deployment entity

        Returns:
            Alignment results if cached, None otherwise
        """
        deploy_dir = self._get_deployment_dir(deployment.id)
        alignment_path = deploy_dir / "cache" / "alignment" / "alignment.json"

        if not alignment_path.exists():
            return None

        data = json.loads(alignment_path.read_text())
        return {
            source_id: AlignmentResult.model_validate(result)
            for source_id, result in data.items()
        }
