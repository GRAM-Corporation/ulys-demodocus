# Wave 3: Merging & Semantic Analysis

Implement transcript merging, semantic analysis, and search indexing.

## Components

### 1. TranscriptMerger (`src/gram_deploy/services/transcript_merger.py`)
Create unified canonical transcript:
- `merge_transcripts()` - Combine multiple source transcripts
- `_detect_duplicates()` - Find overlapping utterances from different sources
- `_resolve_conflicts()` - Pick best version when sources disagree
- `_apply_speaker_mappings()` - Replace raw speaker IDs with person IDs
- `_calculate_canonical_times()` - Convert source times to canonical timeline
- Output list of `CanonicalUtterance` objects
- Save to `deployment/canonical_transcript.json`

### 2. SemanticAnalyzer (`src/gram_deploy/services/semantic_analyzer.py`)
LLM-based extraction using Anthropic API:
- `analyze()` - Full semantic analysis pipeline
- `_extract_events()` - Identify significant moments (decisions, discoveries, problems)
- `_extract_action_items()` - Find tasks, follow-ups, assignments
- `_extract_insights()` - Observations, patterns, recommendations
- `_generate_summary()` - Executive summary of deployment
- Use structured prompts with JSON output parsing
- Save to `deployment/events.json`, `action_items.json`, `insights.json`

### 3. SearchIndexBuilder (`src/gram_deploy/services/search_index.py`)
SQLite FTS5 full-text search:
- `build_index()` - Create/update search database
- `index_utterances()` - Add canonical utterances to FTS
- `index_events()` - Add events with metadata
- `index_insights()` - Add insights
- `search()` - Query with ranking and snippets
- `search_by_speaker()` - Filter by person
- `search_by_timerange()` - Filter by canonical time
- Database at `deployments/search.db`

## Testing
Create `tests/test_wave3.py` with:
- Transcript merging with overlapping sources
- Mock Anthropic API for semantic analysis
- Search index queries and ranking

## Dependencies
- Requires Wave 2 complete (aligned transcripts, speaker mappings)
- Use sample transcript for integration testing
