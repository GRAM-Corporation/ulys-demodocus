# SemanticAnalyzer Service

Implement LLM-based analysis in `src/gram_deploy/services/semantic_analyzer.py`.

## Tasks

1. `analyze(deployment)` -> AnalysisResult:
   - Load canonical transcript
   - Run all extraction pipelines
   - Save results to deployment directory

2. `_extract_events(transcript)` -> list[DeploymentEvent]:
   - Prompt Claude to identify significant moments:
     - Decisions made
     - Problems discovered
     - Solutions proposed
     - Key observations
   - Return structured DeploymentEvent objects
   - Save to `deployment/events.json`

3. `_extract_action_items(transcript)` -> list[ActionItem]:
   - Identify tasks, follow-ups, assignments
   - Extract: description, assignee, priority, due context
   - Save to `deployment/action_items.json`

4. `_extract_insights(transcript, events)` -> list[DeploymentInsight]:
   - Higher-level observations
   - Patterns across the deployment
   - Recommendations
   - Save to `deployment/insights.json`

5. `_generate_summary(transcript, events, insights)` -> str:
   - Executive summary paragraph
   - Key highlights
   - Save to `deployment/summary.md`

Use Anthropic API with structured JSON output.

## Tests
Create `tests/test_semantic_analyzer.py`:
- Mock Anthropic API responses
- Test prompt formatting
- Test response parsing

## Files
- `src/gram_deploy/services/semantic_analyzer.py`
- `tests/test_semantic_analyzer.py`
