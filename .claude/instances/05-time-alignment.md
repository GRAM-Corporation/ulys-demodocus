# TimeAlignmentService

Implement multi-source time synchronization in `src/gram_deploy/services/time_alignment.py`.

## Architecture

**Primary method: Transcript-based alignment.** Uses overlapping speech patterns from transcripts to sync sources. No audio file downloads needed.

```
Source A transcript ──┐
                      ├──→ Find matching phrases ──→ Calculate offset
Source B transcript ──┘
```

## Tasks

1. `align_sources(deployment)` -> dict[source_id, AlignmentResult]:
   - Load raw transcripts for all sources
   - Use first source as reference (canonical_offset_ms = 0)
   - Align others relative to reference
   - Return alignment results with confidence scores

2. `_align_by_transcript(transcript_a: RawTranscript, transcript_b: RawTranscript)` -> tuple[int, float]:
   - Extract utterance windows (text + timestamps)
   - Find matching/similar phrases using fuzzy matching
   - Calculate time offset from best matches
   - Return (offset_ms, confidence)

   Algorithm:
   ```python
   def find_matches(a: RawTranscript, b: RawTranscript) -> list[Match]:
       matches = []
       for seg_a in a.segments:
           for seg_b in b.segments:
               similarity = fuzzy_ratio(seg_a.text, seg_b.text)
               if similarity > 0.85:
                   offset = seg_a.start_ms - seg_b.start_ms
                   matches.append(Match(offset, similarity))
       return matches

   def calculate_offset(matches: list[Match]) -> tuple[int, float]:
       # Cluster offsets, find dominant cluster
       # Return median offset and confidence based on cluster tightness
   ```

3. `_align_by_metadata(source: Source)` -> tuple[int, float]:
   - Parse video file creation timestamps from source.files
   - Calculate offset from reference source
   - Low confidence fallback (0.3-0.5)

4. `calculate_canonical_timeline(alignments: dict)`:
   - Set reference source canonical_offset_ms = 0
   - Apply calculated offsets to other sources
   - Determine canonical_start_time and canonical_end_time for deployment
   - Update source.json files

5. AlignmentResult model:
   ```python
   class AlignmentResult(BaseModel):
       source_id: str
       canonical_offset_ms: int
       confidence: float  # 0.0 - 1.0
       method: str  # "transcript" | "metadata"
       match_count: int  # number of matching phrases found
   ```

## Confidence Scoring

| Method | Confidence Range | When to Use |
|--------|------------------|-------------|
| transcript_match (5+ matches) | 0.85-0.95 | Primary - multiple overlapping phrases |
| transcript_match (2-4 matches) | 0.7-0.85 | Acceptable - some overlap |
| transcript_match (1 match) | 0.5-0.7 | Weak - single phrase match |
| metadata | 0.3-0.5 | Fallback - no transcript overlap |

## Edge Cases

- Single source: canonical_offset_ms = 0, confidence = 1.0
- No transcript overlap: fall back to metadata alignment
- Conflicting matches: use clustering to find consensus

## Tests

Create `tests/test_time_alignment.py`:
- Test transcript matching with known offsets
- Test fuzzy matching threshold tuning
- Test offset clustering algorithm
- Test metadata fallback
- Test single-source case

## Files

- `src/gram_deploy/services/time_alignment.py`
- `tests/test_time_alignment.py`
