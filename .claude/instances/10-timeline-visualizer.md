# TimelineVisualizer Service

Implement interactive HTML timeline in `src/gram_deploy/services/timeline_visualizer.py`.

## Tasks

1. `generate_timeline(deployment)` -> Path:
   - Load canonical transcript, events, action items
   - Render interactive HTML visualization
   - Output to `deployment/timeline.html`

2. Create Jinja2 template `src/gram_deploy/templates/timeline.html`:
   - Horizontal timeline with time axis
   - Speaker lanes (one row per person)
   - Utterance blocks on speaker lanes
   - Event markers with icons by type
   - Action item indicators

3. Interactivity (JavaScript):
   - Zoom in/out on timeline
   - Click utterance to see full text
   - Click event for details popup
   - Time cursor for playback sync

4. Styling:
   - Color-code speakers consistently
   - Event type icons (decision, problem, observation)
   - Priority indicators for action items
   - Responsive layout

5. `_prepare_timeline_data(deployment)`:
   - Convert models to JSON for template
   - Calculate visual positions
   - Group overlapping items

## Tests
Create `tests/test_timeline_visualizer.py`:
- Test data preparation
- Validate HTML output structure

## Files
- `src/gram_deploy/services/timeline_visualizer.py`
- `src/gram_deploy/templates/timeline.html`
- `tests/test_timeline_visualizer.py`
