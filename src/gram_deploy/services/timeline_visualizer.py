"""Timeline Visualizer - generates interactive visualizations of deployments.

Responsible for:
- Interactive HTML timeline with source coverage
- Speaker lanes (one row per person)
- Event markers with type icons
- Action item indicators
- Gantt charts of deployment phases
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

from gram_deploy.models import (
    ActionItem,
    CanonicalUtterance,
    Deployment,
    DeploymentEvent,
    EventType,
    Priority,
    Source,
)


class TimelineVisualizer:
    """Generates timeline visualizations for deployments."""

    # Color palette for speakers
    SPEAKER_COLORS = [
        "#4285F4",  # Blue
        "#EA4335",  # Red
        "#FBBC04",  # Yellow
        "#34A853",  # Green
        "#9C27B0",  # Purple
        "#FF6D00",  # Orange
        "#00BCD4",  # Cyan
        "#E91E63",  # Pink
    ]

    # Event type icons (using Unicode symbols)
    EVENT_ICONS = {
        EventType.DECISION: "\u2714",      # Check mark
        EventType.ISSUE: "\u26A0",         # Warning sign
        EventType.OBSERVATION: "\u2139",   # Info
        EventType.MILESTONE: "\u2605",     # Star
        EventType.DEPLOYMENT_START: "\u25B6",  # Play
        EventType.DEPLOYMENT_END: "\u25A0",    # Stop
        EventType.PHASE_START: "\u25B7",   # Right triangle outline
        EventType.PHASE_END: "\u25B6",     # Right triangle filled
        EventType.ACTION_ITEM: "\u2610",   # Ballot box
        EventType.CUSTOM: "\u25CF",        # Black circle
    }

    # Event type colors
    EVENT_COLORS = {
        EventType.DECISION: "#4CAF50",     # Green
        EventType.ISSUE: "#F44336",        # Red
        EventType.OBSERVATION: "#2196F3",  # Blue
        EventType.MILESTONE: "#FF9800",    # Orange
        EventType.DEPLOYMENT_START: "#4CAF50",
        EventType.DEPLOYMENT_END: "#F44336",
        EventType.PHASE_START: "#9C27B0",
        EventType.PHASE_END: "#9C27B0",
        EventType.ACTION_ITEM: "#FF5722",
        EventType.CUSTOM: "#607D8B",
    }

    # Priority colors for action items
    PRIORITY_COLORS = {
        Priority.CRITICAL: "#F44336",  # Red
        Priority.HIGH: "#FF9800",      # Orange
        Priority.MEDIUM: "#FFEB3B",    # Yellow
        Priority.LOW: "#4CAF50",       # Green
        None: "#9E9E9E",               # Grey
    }

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the visualizer.

        Args:
            data_dir: Root directory for deployment data (default: "deployments")
        """
        self.data_dir = Path(data_dir) if data_dir else Path("deployments")

        # Setup Jinja2 environment
        template_dir = Path(__file__).parent.parent / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate_timeline(
        self,
        deployment: Union[Deployment, str],
        data_dir: Optional[str] = None,
    ) -> Path:
        """Generate interactive HTML timeline for a deployment.

        Loads canonical transcript, events, and action items from disk,
        renders an interactive HTML visualization, and writes to
        deployment/timeline.html.

        Args:
            deployment: Deployment entity or deployment ID string
            data_dir: Optional override for data directory

        Returns:
            Path to the generated timeline.html file
        """
        effective_data_dir = Path(data_dir) if data_dir else self.data_dir

        # Load deployment if ID string provided
        if isinstance(deployment, str):
            deployment = self._load_deployment(deployment, effective_data_dir)

        # Prepare timeline data (loads all dependencies)
        timeline_data = self._prepare_timeline_data(deployment, effective_data_dir)

        # Render HTML
        html_content = self._render_timeline_html(deployment, timeline_data)

        # Write to deployment directory
        deploy_dir = self._get_deployment_dir(deployment.id, effective_data_dir)
        output_path = deploy_dir / "timeline.html"
        output_path.write_text(html_content)

        return output_path

    def _prepare_timeline_data(
        self,
        deployment: Deployment,
        data_dir: Path,
    ) -> dict[str, Any]:
        """Prepare all data needed for timeline visualization.

        Loads and processes:
        - Canonical transcript (utterances)
        - Events
        - Action items
        - Sources
        - People names

        Args:
            deployment: The Deployment entity
            data_dir: Root data directory

        Returns:
            Dictionary with all prepared timeline data
        """
        deploy_dir = self._get_deployment_dir(deployment.id, data_dir)

        # Load utterances
        utterances = self._load_canonical_utterances(deploy_dir)

        # Load events
        events = self._load_events(deploy_dir)

        # Load action items
        action_items = self._load_action_items(deploy_dir)

        # Load sources
        sources = self._load_sources(deployment, data_dir)

        # Load people names
        people_names = self._load_people_names(data_dir)

        # Calculate timeline duration
        duration_ms = self._calculate_duration(deployment, utterances, sources)

        # Group utterances by speaker for speaker lanes
        speaker_lanes = self._group_by_speaker(utterances, people_names)

        # Assign colors to speakers
        speaker_colors = self._assign_speaker_colors(speaker_lanes.keys())

        # Prepare utterance data for JSON
        utterance_data = self._prepare_utterance_data(utterances, people_names)

        # Prepare event data for JSON
        event_data = self._prepare_event_data(events, duration_ms)

        # Prepare action item data for JSON
        action_item_data = self._prepare_action_item_data(action_items, duration_ms)

        # Prepare speaker lane data for JSON
        speaker_lane_data = self._prepare_speaker_lane_data(
            speaker_lanes, speaker_colors, duration_ms
        )

        # Prepare source track data for JSON
        source_track_data = self._prepare_source_track_data(sources, duration_ms)

        return {
            "duration_ms": duration_ms,
            "utterances": utterance_data,
            "events": event_data,
            "action_items": action_item_data,
            "speaker_lanes": speaker_lane_data,
            "speaker_colors": speaker_colors,
            "source_tracks": source_track_data,
            "people_names": people_names,
        }

    def _render_timeline_html(
        self,
        deployment: Deployment,
        timeline_data: dict[str, Any],
    ) -> str:
        """Render timeline HTML using Jinja2 template.

        Falls back to embedded template if template file doesn't exist.

        Args:
            deployment: The Deployment entity
            timeline_data: Prepared timeline data

        Returns:
            Complete HTML document as string
        """
        try:
            template = self.jinja_env.get_template("timeline.html")
            return template.render(
                deployment=deployment,
                **timeline_data,
            )
        except Exception:
            # Fall back to embedded HTML generation
            return self._generate_embedded_html(deployment, timeline_data)

    def _generate_embedded_html(
        self,
        deployment: Deployment,
        timeline_data: dict[str, Any],
    ) -> str:
        """Generate HTML directly without Jinja2 template."""
        duration_ms = timeline_data["duration_ms"]
        speaker_lanes = timeline_data["speaker_lanes"]
        events = timeline_data["events"]
        action_items = timeline_data["action_items"]
        utterances = timeline_data["utterances"]
        source_tracks = timeline_data["source_tracks"]

        # Generate speaker lane HTML
        speaker_lanes_html = self._generate_speaker_lanes_html(speaker_lanes, duration_ms)

        # Generate event markers HTML
        event_markers_html = self._generate_event_markers_html(events, duration_ms)

        # Generate action item markers HTML
        action_items_html = self._generate_action_items_html(action_items, duration_ms)

        # Generate source track HTML
        source_tracks_html = self._generate_source_tracks_html(source_tracks, duration_ms)

        # Generate utterance list HTML
        utterances_html = self._generate_utterances_html(utterances)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Timeline: {deployment.location} - {deployment.date}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }}
        .container {{ display: flex; height: 100vh; }}
        .timeline-panel {{ flex: 2; padding: 20px; overflow: hidden; display: flex; flex-direction: column; }}
        .transcript-panel {{ flex: 1; border-left: 1px solid #333; padding: 20px; overflow-y: auto; background: #16213e; }}
        h1 {{ font-size: 1.5rem; margin-bottom: 10px; }}
        h2 {{ font-size: 1.2rem; margin-bottom: 10px; color: #888; }}
        h3 {{ font-size: 1rem; margin: 10px 0 5px; color: #aaa; }}
        .timeline-container {{ flex: 1; position: relative; overflow-x: auto; overflow-y: auto; }}
        .timeline {{ position: relative; min-width: 100%; min-height: 100%; }}
        .time-axis {{ position: sticky; top: 0; left: 0; right: 0; height: 30px; background: #0f3460; border-bottom: 1px solid #333; z-index: 10; }}
        .time-label {{ position: absolute; font-size: 12px; color: #888; transform: translateX(-50%); top: 8px; }}

        /* Speaker lanes */
        .speaker-section {{ margin-top: 35px; }}
        .speaker-lane {{ position: relative; height: 50px; margin: 5px 0; background: #16213e; border-radius: 4px; border-left: 4px solid; }}
        .speaker-label {{ position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-size: 12px; width: 120px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; z-index: 5; }}
        .utterance-block {{ position: absolute; height: 30px; top: 10px; border-radius: 3px; opacity: 0.8; cursor: pointer; min-width: 4px; left: 130px; }}
        .utterance-block:hover {{ opacity: 1; transform: scaleY(1.1); }}

        /* Event markers */
        .events-section {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #333; }}
        .events-track {{ position: relative; height: 60px; margin-left: 130px; }}
        .event-marker {{ position: absolute; cursor: pointer; text-align: center; transform: translateX(-50%); }}
        .event-icon {{ font-size: 20px; display: block; }}
        .event-label {{ font-size: 10px; color: #aaa; max-width: 80px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .event-tooltip {{ position: absolute; background: #0f3460; padding: 10px; border-radius: 4px; font-size: 12px; z-index: 100; display: none; min-width: 200px; max-width: 300px; left: 50%; transform: translateX(-50%); top: 45px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .event-marker:hover .event-tooltip {{ display: block; }}
        .event-tooltip h4 {{ margin-bottom: 5px; }}
        .event-tooltip p {{ color: #aaa; }}

        /* Action item markers */
        .action-items-section {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #333; }}
        .action-items-track {{ position: relative; height: 40px; margin-left: 130px; }}
        .action-marker {{ position: absolute; cursor: pointer; transform: translateX(-50%); }}
        .action-icon {{ width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; }}
        .action-tooltip {{ position: absolute; background: #0f3460; padding: 10px; border-radius: 4px; font-size: 12px; z-index: 100; display: none; min-width: 200px; max-width: 300px; left: 50%; transform: translateX(-50%); top: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .action-marker:hover .action-tooltip {{ display: block; }}
        .priority-badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 5px; }}

        /* Source tracks */
        .sources-section {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #333; }}
        .source-track {{ position: relative; height: 35px; margin: 3px 0; background: #16213e; border-radius: 4px; }}
        .source-label {{ position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-size: 11px; width: 120px; color: #888; }}
        .source-coverage {{ position: absolute; height: 25px; top: 5px; border-radius: 3px; opacity: 0.6; cursor: pointer; left: 130px; }}
        .source-coverage:hover {{ opacity: 0.9; }}

        /* Transcript panel */
        .utterance {{ padding: 10px; border-bottom: 1px solid #333; cursor: pointer; transition: background 0.2s; }}
        .utterance:hover {{ background: #0f3460; }}
        .utterance.active {{ background: #1a4980; border-left: 3px solid #4285F4; }}
        .utterance-time {{ font-size: 11px; color: #888; }}
        .utterance-speaker {{ font-weight: bold; margin-left: 10px; }}
        .utterance-text {{ margin-top: 5px; line-height: 1.4; }}

        /* Controls */
        .controls {{ display: flex; gap: 10px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; }}
        .controls button {{ padding: 8px 16px; background: #0f3460; border: none; color: #eee; border-radius: 4px; cursor: pointer; }}
        .controls button:hover {{ background: #1a4980; }}
        .controls .time-display {{ font-family: monospace; background: #0f3460; padding: 8px 16px; border-radius: 4px; }}

        /* Playhead */
        .playhead {{ position: absolute; width: 2px; background: #fff; z-index: 50; pointer-events: none; top: 0; bottom: 0; left: 130px; }}

        /* Legend */
        .legend {{ display: flex; gap: 15px; margin: 10px 0; flex-wrap: wrap; font-size: 12px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
        .legend-icon {{ font-size: 14px; }}

        /* Responsive */
        @media (max-width: 900px) {{
            .container {{ flex-direction: column; }}
            .timeline-panel {{ flex: none; height: 60vh; }}
            .transcript-panel {{ flex: none; height: 40vh; border-left: none; border-top: 1px solid #333; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="timeline-panel">
            <h1>{deployment.location} - {deployment.date}</h1>
            <div class="controls">
                <button onclick="zoomIn()">Zoom In</button>
                <button onclick="zoomOut()">Zoom Out</button>
                <button onclick="resetZoom()">Reset</button>
                <div class="time-display" id="timeDisplay">00:00:00</div>
            </div>
            <div class="legend">
                <div class="legend-item"><span class="legend-icon">{self.EVENT_ICONS[EventType.DECISION]}</span> Decision</div>
                <div class="legend-item"><span class="legend-icon">{self.EVENT_ICONS[EventType.ISSUE]}</span> Issue</div>
                <div class="legend-item"><span class="legend-icon">{self.EVENT_ICONS[EventType.OBSERVATION]}</span> Observation</div>
                <div class="legend-item"><span class="legend-icon">{self.EVENT_ICONS[EventType.MILESTONE]}</span> Milestone</div>
            </div>
            <div class="timeline-container" id="timelineContainer">
                <div class="timeline" id="timeline">
                    <div class="time-axis" id="timeAxis"></div>

                    <div class="speaker-section">
                        <h3>Speaker Activity</h3>
                        {speaker_lanes_html}
                    </div>

                    <div class="events-section">
                        <h3>Events</h3>
                        <div class="events-track" id="eventsTrack">
                            {event_markers_html}
                        </div>
                    </div>

                    <div class="action-items-section">
                        <h3>Action Items</h3>
                        <div class="action-items-track" id="actionItemsTrack">
                            {action_items_html}
                        </div>
                    </div>

                    <div class="sources-section">
                        <h3>Source Coverage</h3>
                        {source_tracks_html}
                    </div>

                    <div class="playhead" id="playhead" style="display: none;"></div>
                </div>
            </div>
        </div>
        <div class="transcript-panel">
            <h2>Transcript</h2>
            <div id="transcript">
                {utterances_html}
            </div>
        </div>
    </div>
    <script>
        const DURATION_MS = {duration_ms};
        const TIMELINE_DATA = {json.dumps(timeline_data)};
        let zoom = 1;
        let currentTime = 0;

        function formatTime(ms) {{
            const seconds = Math.floor(ms / 1000);
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = seconds % 60;
            return `${{h.toString().padStart(2, '0')}}:${{m.toString().padStart(2, '0')}}:${{s.toString().padStart(2, '0')}}`;
        }}

        function updateTimeAxis() {{
            const axis = document.getElementById('timeAxis');
            const timeline = document.getElementById('timeline');
            const width = timeline.offsetWidth - 130; // Account for label area
            const interval = Math.max(60000, Math.floor(DURATION_MS / (width / 100)));
            axis.innerHTML = '';
            for (let t = 0; t <= DURATION_MS; t += interval) {{
                const label = document.createElement('div');
                label.className = 'time-label';
                label.style.left = (130 + (t / DURATION_MS) * width) + 'px';
                label.textContent = formatTime(t);
                axis.appendChild(label);
            }}
        }}

        function zoomIn() {{
            zoom *= 1.5;
            document.getElementById('timeline').style.width = (zoom * 100) + '%';
            updateTimeAxis();
        }}

        function zoomOut() {{
            zoom = Math.max(1, zoom / 1.5);
            document.getElementById('timeline').style.width = (zoom * 100) + '%';
            updateTimeAxis();
        }}

        function resetZoom() {{
            zoom = 1;
            document.getElementById('timeline').style.width = '100%';
            updateTimeAxis();
        }}

        function seekTo(ms) {{
            currentTime = ms;
            document.getElementById('timeDisplay').textContent = formatTime(ms);

            const playhead = document.getElementById('playhead');
            const timeline = document.getElementById('timeline');
            const width = timeline.offsetWidth - 130;
            playhead.style.display = 'block';
            playhead.style.left = (130 + (ms / DURATION_MS) * width) + 'px';

            // Highlight active utterance and scroll to it
            const utterances = document.querySelectorAll('.utterance');
            let targetUtterance = null;
            for (const u of utterances) {{
                const start = parseInt(u.dataset.start);
                const end = parseInt(u.dataset.end);
                if (ms >= start && ms < end) {{
                    u.classList.add('active');
                    targetUtterance = u;
                }} else {{
                    u.classList.remove('active');
                }}
            }}
            if (targetUtterance) {{
                targetUtterance.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        // Click handlers for utterance blocks
        document.querySelectorAll('.utterance-block').forEach(el => {{
            el.addEventListener('click', () => {{
                seekTo(parseInt(el.dataset.start));
            }});
        }});

        // Click handlers for transcript utterances
        document.querySelectorAll('.utterance').forEach(el => {{
            el.addEventListener('click', () => {{
                seekTo(parseInt(el.dataset.start));
            }});
        }});

        // Click on timeline to seek
        document.getElementById('timelineContainer').addEventListener('click', (e) => {{
            if (e.target.closest('.event-marker') || e.target.closest('.action-marker') ||
                e.target.closest('.utterance-block') || e.target.closest('.source-coverage')) return;

            const timeline = document.getElementById('timeline');
            const rect = timeline.getBoundingClientRect();
            const x = e.clientX - rect.left - 130;
            if (x < 0) return;
            const width = timeline.offsetWidth - 130;
            const time = (x / width) * DURATION_MS;
            seekTo(Math.max(0, Math.min(DURATION_MS, Math.floor(time))));
        }});

        // Initialize
        updateTimeAxis();
        window.addEventListener('resize', updateTimeAxis);
    </script>
</body>
</html>"""

        return html

    def _generate_speaker_lanes_html(
        self,
        speaker_lanes: list[dict],
        duration_ms: int,
    ) -> str:
        """Generate HTML for speaker lanes."""
        if not speaker_lanes:
            return '<p style="color: #666; padding: 10px;">No speaker data available</p>'

        html_parts = []
        for lane in speaker_lanes:
            speaker_name = lane["speaker_name"]
            color = lane["color"]
            utterances = lane["utterances"]

            # Generate utterance blocks
            blocks = []
            for u in utterances:
                left_pct = (u["start_ms"] / duration_ms) * 100
                width_pct = max(0.2, (u["duration_ms"] / duration_ms) * 100)
                blocks.append(f'''
                    <div class="utterance-block"
                         style="left: calc(130px + {left_pct}%); width: {width_pct}%; background: {color};"
                         data-start="{u['start_ms']}" data-end="{u['end_ms']}"
                         title="{self._escape_html(u['text'][:50])}...">
                    </div>
                ''')

            html_parts.append(f'''
                <div class="speaker-lane" style="border-left-color: {color};">
                    <div class="speaker-label" style="color: {color};">{self._escape_html(speaker_name)}</div>
                    {''.join(blocks)}
                </div>
            ''')

        return '\n'.join(html_parts)

    def _generate_event_markers_html(
        self,
        events: list[dict],
        duration_ms: int,
    ) -> str:
        """Generate HTML for event markers."""
        if not events:
            return '<p style="color: #666; padding: 10px;">No events recorded</p>'

        markers = []
        for event in events:
            left_pct = (event["time_ms"] / duration_ms) * 100
            icon = event["icon"]
            color = event["color"]
            title = event["title"]
            description = event.get("description", "")
            event_type = event["type"]

            markers.append(f'''
                <div class="event-marker" style="left: {left_pct}%;">
                    <span class="event-icon" style="color: {color};">{icon}</span>
                    <span class="event-label">{self._escape_html(title[:15])}</span>
                    <div class="event-tooltip">
                        <h4>{self._escape_html(title)}</h4>
                        <p><strong>Type:</strong> {event_type}</p>
                        <p><strong>Time:</strong> {self._format_time(event["time_ms"])}</p>
                        {f'<p>{self._escape_html(description)}</p>' if description else ''}
                    </div>
                </div>
            ''')

        return '\n'.join(markers)

    def _generate_action_items_html(
        self,
        action_items: list[dict],
        duration_ms: int,
    ) -> str:
        """Generate HTML for action item markers."""
        if not action_items:
            return '<p style="color: #666; padding: 10px;">No action items recorded</p>'

        markers = []
        for item in action_items:
            left_pct = (item["time_ms"] / duration_ms) * 100
            color = item["priority_color"]
            description = item["description"]
            priority = item.get("priority", "unset")
            mentioned_by = item.get("mentioned_by", "Unknown")

            markers.append(f'''
                <div class="action-marker" style="left: {left_pct}%;">
                    <div class="action-icon" style="background: {color};">{self.EVENT_ICONS[EventType.ACTION_ITEM]}</div>
                    <div class="action-tooltip">
                        <p><strong>{self._escape_html(description)}</strong></p>
                        <p>Priority: <span class="priority-badge" style="background: {color};">{priority}</span></p>
                        <p>Mentioned by: {self._escape_html(mentioned_by)}</p>
                        <p>Time: {self._format_time(item["time_ms"])}</p>
                    </div>
                </div>
            ''')

        return '\n'.join(markers)

    def _generate_source_tracks_html(
        self,
        source_tracks: list[dict],
        duration_ms: int,
    ) -> str:
        """Generate HTML for source coverage tracks."""
        if not source_tracks:
            return '<p style="color: #666; padding: 10px;">No source data available</p>'

        html_parts = []
        colors = ["#4285F4", "#EA4335", "#FBBC04", "#34A853", "#9C27B0", "#FF6D00"]

        for i, track in enumerate(source_tracks):
            color = colors[i % len(colors)]
            label = track["label"]
            segments = track["segments"]

            # Generate coverage blocks
            blocks = []
            for seg in segments:
                left_pct = (seg["start_ms"] / duration_ms) * 100
                width_pct = max(0.2, (seg["duration_ms"] / duration_ms) * 100)
                blocks.append(f'''
                    <div class="source-coverage"
                         style="left: calc(130px + {left_pct}%); width: {width_pct}%; background: {color};"
                         data-start="{seg['start_ms']}" data-end="{seg['end_ms']}"
                         title="{seg.get('filename', '')}">
                    </div>
                ''')

            html_parts.append(f'''
                <div class="source-track">
                    <div class="source-label">{self._escape_html(label)}</div>
                    {''.join(blocks)}
                </div>
            ''')

        return '\n'.join(html_parts)

    def _generate_utterances_html(self, utterances: list[dict]) -> str:
        """Generate HTML for utterance list in transcript panel."""
        if not utterances:
            return '<p style="color: #666; padding: 10px;">No transcript available</p>'

        items = []
        for u in utterances[:500]:  # Limit for performance
            time_str = self._format_time(u["start_ms"])
            speaker = u.get("speaker_name", "Unknown")
            text = u.get("text", "")
            color = u.get("speaker_color", "#4285F4")

            items.append(f'''
                <div class="utterance" data-start="{u['start_ms']}" data-end="{u['end_ms']}">
                    <span class="utterance-time">{time_str}</span>
                    <span class="utterance-speaker" style="color: {color};">{self._escape_html(speaker)}</span>
                    <div class="utterance-text">{self._escape_html(text)}</div>
                </div>
            ''')

        return '\n'.join(items)

    def _group_by_speaker(
        self,
        utterances: list[CanonicalUtterance],
        people_names: dict[str, str],
    ) -> dict[str, list[CanonicalUtterance]]:
        """Group utterances by speaker."""
        by_speaker: dict[str, list[CanonicalUtterance]] = {}
        for u in utterances:
            speaker_name = people_names.get(u.speaker_id, u.speaker_id or "Unknown")
            if speaker_name not in by_speaker:
                by_speaker[speaker_name] = []
            by_speaker[speaker_name].append(u)
        return by_speaker

    def _assign_speaker_colors(self, speakers: list[str]) -> dict[str, str]:
        """Assign colors to speakers consistently."""
        colors = {}
        speaker_list = sorted(speakers)  # Sort for consistent coloring
        for i, speaker in enumerate(speaker_list):
            colors[speaker] = self.SPEAKER_COLORS[i % len(self.SPEAKER_COLORS)]
        return colors

    def _prepare_utterance_data(
        self,
        utterances: list[CanonicalUtterance],
        people_names: dict[str, str],
    ) -> list[dict]:
        """Prepare utterance data for JSON serialization."""
        speaker_colors = self._assign_speaker_colors(
            set(people_names.get(u.speaker_id, u.speaker_id or "Unknown") for u in utterances)
        )

        return [
            {
                "id": u.id,
                "text": u.text,
                "start_ms": u.canonical_start_ms,
                "end_ms": u.canonical_end_ms,
                "duration_ms": u.duration_ms,
                "speaker_id": u.speaker_id,
                "speaker_name": people_names.get(u.speaker_id, u.speaker_id or "Unknown"),
                "speaker_color": speaker_colors.get(
                    people_names.get(u.speaker_id, u.speaker_id or "Unknown"),
                    "#4285F4"
                ),
            }
            for u in utterances
        ]

    def _prepare_event_data(
        self,
        events: list[DeploymentEvent],
        duration_ms: int,
    ) -> list[dict]:
        """Prepare event data for JSON serialization."""
        result = []
        for e in events:
            # Handle both enum and string values for event_type
            event_type = e.event_type
            event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
            event_type_enum = EventType(event_type_str) if isinstance(event_type, str) else event_type

            # Handle severity
            severity = e.severity
            severity_str = severity.value if hasattr(severity, 'value') else severity if severity else None

            result.append({
                "id": e.id,
                "type": event_type_str,
                "title": e.title,
                "description": e.description,
                "time_ms": e.canonical_time_ms,
                "icon": self.EVENT_ICONS.get(event_type_enum, "\u25CF"),
                "color": self.EVENT_COLORS.get(event_type_enum, "#607D8B"),
                "severity": severity_str,
            })
        return result

    def _prepare_action_item_data(
        self,
        action_items: list[ActionItem],
        duration_ms: int,
    ) -> list[dict]:
        """Prepare action item data for JSON serialization."""
        result = []
        for item in action_items:
            # Handle both enum and string values for priority
            priority = item.priority
            priority_str = priority.value if hasattr(priority, 'value') else priority if priority else "unset"
            priority_enum = Priority(priority_str) if isinstance(priority, str) and priority != "unset" else priority

            # Handle status
            status = item.status
            status_str = status.value if hasattr(status, 'value') else str(status)

            result.append({
                "id": item.id,
                "description": item.description,
                "time_ms": item.canonical_time_ms,
                "priority": priority_str if priority_str else "unset",
                "priority_color": self.PRIORITY_COLORS.get(priority_enum, "#9E9E9E"),
                "mentioned_by": item.mentioned_by,
                "assigned_to": item.assigned_to,
                "status": status_str,
            })
        return result

    def _prepare_speaker_lane_data(
        self,
        speaker_lanes: dict[str, list[CanonicalUtterance]],
        speaker_colors: dict[str, str],
        duration_ms: int,
    ) -> list[dict]:
        """Prepare speaker lane data for JSON serialization."""
        lanes = []
        for speaker_name in sorted(speaker_lanes.keys()):
            utterances = speaker_lanes[speaker_name]
            lanes.append({
                "speaker_name": speaker_name,
                "color": speaker_colors.get(speaker_name, "#4285F4"),
                "utterances": [
                    {
                        "start_ms": u.canonical_start_ms,
                        "end_ms": u.canonical_end_ms,
                        "duration_ms": u.duration_ms,
                        "text": u.text,
                    }
                    for u in utterances
                ],
            })
        return lanes

    def _prepare_source_track_data(
        self,
        sources: list[Source],
        duration_ms: int,
    ) -> list[dict]:
        """Prepare source track data for JSON serialization."""
        tracks = []
        for source in sources:
            segments = []
            for f in source.files:
                start_ms = source.canonical_offset_ms + f.start_offset_ms
                segments.append({
                    "start_ms": start_ms,
                    "end_ms": start_ms + int(f.duration_seconds * 1000),
                    "duration_ms": int(f.duration_seconds * 1000),
                    "filename": f.filename,
                })

            # Handle both enum and string values for device_type
            device_type = source.device_type
            device_type_str = device_type.value if hasattr(device_type, 'value') else str(device_type)

            tracks.append({
                "source_id": source.id,
                "label": f"{device_type_str} {source.device_number}",
                "segments": segments,
            })
        return tracks

    def _calculate_duration(
        self,
        deployment: Deployment,
        utterances: list[CanonicalUtterance],
        sources: list[Source],
    ) -> int:
        """Calculate total timeline duration in milliseconds."""
        # Try from deployment times
        if deployment.canonical_start_time and deployment.canonical_end_time:
            return int(
                (deployment.canonical_end_time - deployment.canonical_start_time).total_seconds() * 1000
            )

        # Try from utterances
        if utterances:
            return max(u.canonical_end_ms for u in utterances)

        # Try from sources
        if sources:
            return max(
                s.canonical_offset_ms + int(s.total_duration * 1000)
                for s in sources
            )

        return 3600000  # Default 1 hour

    # File I/O methods

    def _get_deployment_dir(self, deployment_id: str, data_dir: Path) -> Path:
        """Get the directory path for a deployment."""
        dir_name = deployment_id.replace(":", "_")
        return data_dir / dir_name

    def _load_deployment(self, deployment_id: str, data_dir: Path) -> Deployment:
        """Load a deployment from disk."""
        deploy_dir = self._get_deployment_dir(deployment_id, data_dir)
        deployment_path = deploy_dir / "deployment.json"

        if not deployment_path.exists():
            raise FileNotFoundError(f"Deployment not found: {deployment_id}")

        data = json.loads(deployment_path.read_text())
        return Deployment.model_validate(data)

    def _load_canonical_utterances(self, deploy_dir: Path) -> list[CanonicalUtterance]:
        """Load canonical utterances from deployment directory."""
        transcript_path = deploy_dir / "canonical" / "transcript.json"

        if not transcript_path.exists():
            return []

        data = json.loads(transcript_path.read_text())

        # Handle both array format and object with utterances key
        if isinstance(data, list):
            utterances_data = data
        else:
            utterances_data = data.get("utterances", [])

        return [CanonicalUtterance.model_validate(u) for u in utterances_data]

    def _load_events(self, deploy_dir: Path) -> list[DeploymentEvent]:
        """Load events from deployment directory."""
        events_path = deploy_dir / "events.json"

        if not events_path.exists():
            return []

        data = json.loads(events_path.read_text())
        return [DeploymentEvent.model_validate(e) for e in data]

    def _load_action_items(self, deploy_dir: Path) -> list[ActionItem]:
        """Load action items from deployment directory."""
        items_path = deploy_dir / "action_items.json"

        if not items_path.exists():
            return []

        data = json.loads(items_path.read_text())
        return [ActionItem.model_validate(item) for item in data]

    def _load_sources(self, deployment: Deployment, data_dir: Path) -> list[Source]:
        """Load sources for a deployment."""
        sources = []
        deploy_dir = self._get_deployment_dir(deployment.id, data_dir)

        for source_id in deployment.sources:
            # Extract device part from source ID
            parts = source_id.replace("source:", "").split("/")
            if len(parts) == 2:
                device_part = parts[1]
                source_path = deploy_dir / "sources" / device_part / "source.json"
                if source_path.exists():
                    data = json.loads(source_path.read_text())
                    sources.append(Source.model_validate(data))

        return sources

    def _load_people_names(self, data_dir: Path) -> dict[str, str]:
        """Load people names mapping from registry."""
        registry_path = data_dir / "people.json"

        if not registry_path.exists():
            return {}

        data = json.loads(registry_path.read_text())
        people_names = {}

        for person_data in data.get("people", []):
            person_id = person_data.get("id")
            name = person_data.get("name")
            if person_id and name:
                people_names[person_id] = name

        return people_names

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS."""
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    # Legacy methods for backward compatibility

    def generate_timeline_html(
        self,
        deployment: Deployment,
        sources: list[Source],
        utterances: list[CanonicalUtterance],
        events: list[DeploymentEvent],
        people_names: Optional[dict[str, str]] = None,
        action_items: Optional[list[ActionItem]] = None,
    ) -> str:
        """Generate interactive HTML timeline (legacy method).

        Args:
            deployment: The Deployment entity
            sources: Sources to show
            utterances: Utterances for transcript display
            events: Events to mark on timeline
            people_names: Optional mapping of person_id -> display name
            action_items: Optional action items to display

        Returns:
            Complete HTML document as string
        """
        people_names = people_names or {}
        action_items = action_items or []

        # Calculate timeline duration
        duration_ms = self._calculate_duration(deployment, utterances, sources)

        # Prepare timeline data
        speaker_lanes = self._group_by_speaker(utterances, people_names)
        speaker_colors = self._assign_speaker_colors(speaker_lanes.keys())

        timeline_data = {
            "duration_ms": duration_ms,
            "utterances": self._prepare_utterance_data(utterances, people_names),
            "events": self._prepare_event_data(events, duration_ms),
            "action_items": self._prepare_action_item_data(action_items, duration_ms),
            "speaker_lanes": self._prepare_speaker_lane_data(
                speaker_lanes, speaker_colors, duration_ms
            ),
            "speaker_colors": speaker_colors,
            "source_tracks": self._prepare_source_track_data(sources, duration_ms),
            "people_names": people_names,
        }

        return self._generate_embedded_html(deployment, timeline_data)

    def generate_gantt_chart(
        self,
        deployment: Deployment,
        events: list[DeploymentEvent],
    ) -> str:
        """Generate Gantt-style SVG chart of deployment phases."""
        # Filter to phase events - handle both enum and string values
        def get_event_type_str(e):
            event_type = e.event_type
            return event_type.value if hasattr(event_type, 'value') else str(event_type)

        phases = [e for e in events if get_event_type_str(e).startswith("phase_")]

        # SVG dimensions
        width = 800
        height = max(200, len(phases) * 50 + 100)

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
            <style>
                .phase-bar {{ fill: #4285F4; }}
                .phase-label {{ font-size: 12px; fill: #333; }}
                .time-label {{ font-size: 10px; fill: #888; }}
            </style>
            <rect width="100%" height="100%" fill="#fff"/>
        """

        # Add phase bars
        y = 50
        for event in phases:
            svg += f"""
                <rect x="100" y="{y}" width="200" height="30" class="phase-bar" rx="4"/>
                <text x="90" y="{y + 20}" class="phase-label" text-anchor="end">{event.title}</text>
            """
            y += 50

        svg += "</svg>"
        return svg

    def generate_speaker_timeline(
        self,
        deployment: Deployment,
        utterances: list[CanonicalUtterance],
        people_names: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate visualization of speaker activity over time."""
        people_names = people_names or {}

        # Group utterances by speaker
        by_speaker = self._group_by_speaker(utterances, people_names)

        # Calculate duration
        duration_ms = max(u.canonical_end_ms for u in utterances) if utterances else 3600000

        # Generate SVG
        width = 1000
        row_height = 40
        height = len(by_speaker) * row_height + 60

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
            <style>
                .speaker-label {{ font-size: 12px; fill: #333; }}
                .utterance-block {{ opacity: 0.8; }}
            </style>
            <rect width="100%" height="100%" fill="#f5f5f5"/>
        """

        y = 40
        speaker_colors = self._assign_speaker_colors(by_speaker.keys())
        for speaker, speaker_utterances in sorted(by_speaker.items()):
            color = speaker_colors.get(speaker, self.SPEAKER_COLORS[0])

            svg += f'<text x="10" y="{y + 25}" class="speaker-label">{speaker}</text>'

            for u in speaker_utterances:
                x = 150 + (u.canonical_start_ms / duration_ms) * (width - 160)
                w = max(2, (u.duration_ms / duration_ms) * (width - 160))
                svg += f'<rect x="{x}" y="{y + 5}" width="{w}" height="30" fill="{color}" class="utterance-block" rx="2"/>'

            y += row_height

        svg += "</svg>"
        return svg
