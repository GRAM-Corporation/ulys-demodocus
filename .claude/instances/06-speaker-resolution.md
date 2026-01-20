# SpeakerResolutionService

Implement speaker identification in `src/gram_deploy/services/speaker_resolution.py`.

## Tasks

1. `resolve_speakers(deployment)` -> list[SpeakerMapping]:
   - Load raw transcripts from all sources
   - Load known people from `deployments/people.json`
   - Match raw speaker IDs to person IDs
   - Return mappings with confidence scores

2. `_match_by_voice_embedding(speaker_audio, person)`:
   - Extract voice embedding from speaker segments
   - Compare with stored person voice_embedding
   - Use resemblyzer or speechbrain for comparison
   - Return similarity score

3. `_match_by_context(transcript, person)`:
   - Look for name mentions near speaker turns
   - Check for role-specific language ("as CTO...", "I'll handle the code...")
   - Match against person aliases

4. `_match_by_pattern(speakers, people)`:
   - Analyze speaker frequency and patterns
   - Match expected team composition
   - Use deployment.team_members as hints

5. Save mappings:
   - Write to `{source_path}/speaker_mappings.json`
   - Include raw_speaker_id, person_id, confidence, method

Reference: `deployments/people.json` has Damion (CTO) and Chu

## Tests
Create `tests/test_speaker_resolution.py`:
- Test context matching with sample transcript
- Test pattern analysis
- Mock voice embedding comparison

## Files
- `src/gram_deploy/services/speaker_resolution.py`
- `tests/test_speaker_resolution.py`
