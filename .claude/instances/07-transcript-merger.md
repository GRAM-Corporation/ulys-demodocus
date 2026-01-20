# TranscriptMerger Service

Implement canonical transcript creation in `src/gram_deploy/services/transcript_merger.py`.

## Tasks

1. `merge_transcripts(deployment)` -> list[CanonicalUtterance]:
   - Load all source raw_transcripts
   - Load speaker_mappings for each source
   - Load time alignments (canonical_offset_ms)
   - Merge into single canonical timeline

2. `_convert_to_canonical_time(utterance, source)`:
   - Apply source.canonical_offset_ms
   - Return utterance with canonical start/end times

3. `_detect_duplicates(utterances)`:
   - Find overlapping utterances from different sources
   - Same speaker + similar text + overlapping time = duplicate
   - Use text similarity threshold (e.g., 0.85)

4. `_resolve_conflicts(duplicates)`:
   - Pick best version based on:
     - Higher transcription confidence
     - Better audio quality source
     - More complete text
   - Mark others as alternates

5. `_apply_speaker_mappings(utterances, mappings)`:
   - Replace raw speaker IDs with person IDs
   - Handle unmapped speakers as "unknown_N"

6. Output:
   - Sort by canonical_start_ms
   - Write to `deployment/canonical_transcript.json`

## Tests
Create `tests/test_transcript_merger.py`:
- Test time conversion
- Test duplicate detection
- Test conflict resolution

## Files
- `src/gram_deploy/services/transcript_merger.py`
- `tests/test_transcript_merger.py`
