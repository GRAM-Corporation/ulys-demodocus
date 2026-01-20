# TimeAlignmentService

Implement multi-source time synchronization in `src/gram_deploy/services/time_alignment.py`.

## Tasks

1. `align_sources(deployment)` -> dict[source_id, AlignmentResult]:
   - Process all sources in deployment
   - Calculate canonical_offset_ms for each
   - Return alignment results with confidence scores

2. `_align_by_audio_fingerprint(source_a, source_b)`:
   - Generate chromaprint fingerprints for both audio files
   - Cross-correlate to find time offset
   - Return offset in milliseconds and confidence

3. `_align_by_transcript(transcript_a, transcript_b)`:
   - Find matching phrases between transcripts
   - Calculate time offset from matched segments
   - Lower confidence than audio fingerprint

4. `_align_by_metadata(source)`:
   - Use video file timestamps
   - Fallback when other methods fail
   - Lowest confidence

5. `calculate_canonical_timeline(alignments)`:
   - Determine canonical start time (earliest source)
   - Set canonical_offset_ms for each source
   - Update source.json files

6. Confidence scoring:
   - audio_fingerprint: 0.9-1.0
   - transcript_match: 0.7-0.9
   - metadata: 0.3-0.5

## Tests
Create `tests/test_time_alignment.py`:
- Test with synthetic audio samples
- Test transcript matching algorithm
- Test confidence scoring

## Files
- `src/gram_deploy/services/time_alignment.py`
- `tests/test_time_alignment.py`
