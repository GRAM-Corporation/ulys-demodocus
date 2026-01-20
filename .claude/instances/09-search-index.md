# SearchIndexBuilder Service

Implement full-text search in `src/gram_deploy/services/search_index.py`.

## Tasks

1. `build_index(deployment)`:
   - Create/update SQLite database with FTS5
   - Index all searchable content
   - Database at `deployments/search.db`

2. Schema:
   ```sql
   CREATE VIRTUAL TABLE utterances USING fts5(
     deployment_id, source_id, speaker_id, text,
     canonical_start_ms, canonical_end_ms
   );
   CREATE VIRTUAL TABLE events USING fts5(
     deployment_id, event_type, description, canonical_time_ms
   );
   CREATE VIRTUAL TABLE insights USING fts5(
     deployment_id, category, content
   );
   ```

3. `index_utterances(deployment, utterances)`:
   - Insert canonical utterances into FTS table

4. `index_events(deployment, events)`:
   - Insert deployment events

5. `search(query, deployment_id=None)` -> list[SearchResult]:
   - Full-text search across all tables
   - Return ranked results with snippets
   - Include source location for playback

6. `search_by_speaker(query, person_id)`:
   - Filter results by speaker

7. `search_by_timerange(query, start_ms, end_ms)`:
   - Filter by canonical time range

## Tests
Create `tests/test_search_index.py`:
- Test index creation
- Test search queries
- Test filtering

## Files
- `src/gram_deploy/services/search_index.py`
- `tests/test_search_index.py`
