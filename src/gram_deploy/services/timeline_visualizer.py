"""Timeline Visualizer - generates interactive visualizations of deployments.

Responsible for:
- Interactive HTML timeline with source coverage
- Gantt charts of deployment phases
- Speaker activity timeline
"""

from typing import Optional

from gram_deploy.models import (
    CanonicalUtterance,
    Deployment,
    DeploymentEvent,
    Source,
)


class TimelineVisualizer:
    """Generates timeline visualizations for deployments."""

    # Color palette for sources
    COLORS = [
        "#4285F4",  # Blue
        "#EA4335",  # Red
        "#FBBC04",  # Yellow
        "#34A853",  # Green
        "#9C27B0",  # Purple
        "#FF6D00",  # Orange
        "#00BCD4",  # Cyan
        "#E91E63",  # Pink
    ]

    def __init__(self):
        """Initialize the visualizer."""
        pass

    def generate_timeline_html(
        self,
        deployment: Deployment,
        sources: list[Source],
        utterances: list[CanonicalUtterance],
        events: list[DeploymentEvent],
        people_names: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate interactive HTML timeline.

        Args:
            deployment: The Deployment entity
            sources: Sources to show
            utterances: Utterances for transcript display
            events: Events to mark on timeline
            people_names: Optional mapping of person_id -> display name

        Returns:
            Complete HTML document as string
        """
        people_names = people_names or {}

        # Calculate timeline dimensions
        if deployment.canonical_start_time and deployment.canonical_end_time:
            duration_ms = int(
                (deployment.canonical_end_time - deployment.canonical_start_time).total_seconds() * 1000
            )
        else:
            duration_ms = max(
                (s.canonical_offset_ms + int(s.total_duration * 1000))
                for s in sources
            ) if sources else 3600000

        # Generate source tracks
        source_tracks = self._generate_source_tracks(sources, duration_ms)

        # Generate event markers
        event_markers = self._generate_event_markers(events, duration_ms)

        # Generate utterance data for transcript panel
        utterance_data = self._generate_utterance_data(utterances, people_names)

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
        .timeline-panel {{ flex: 2; padding: 20px; overflow: hidden; }}
        .transcript-panel {{ flex: 1; border-left: 1px solid #333; padding: 20px; overflow-y: auto; background: #16213e; }}
        h1 {{ font-size: 1.5rem; margin-bottom: 10px; }}
        h2 {{ font-size: 1.2rem; margin-bottom: 10px; color: #888; }}
        .timeline-container {{ position: relative; height: calc(100% - 100px); overflow-x: auto; overflow-y: hidden; }}
        .timeline {{ position: relative; min-width: 100%; height: 100%; }}
        .time-axis {{ position: absolute; top: 0; left: 0; right: 0; height: 30px; background: #0f3460; border-bottom: 1px solid #333; }}
        .time-label {{ position: absolute; font-size: 12px; color: #888; transform: translateX(-50%); }}
        .source-track {{ position: relative; height: 40px; margin: 5px 0; background: #16213e; border-radius: 4px; }}
        .source-label {{ position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-size: 12px; width: 100px; }}
        .source-coverage {{ position: absolute; height: 30px; top: 5px; border-radius: 3px; opacity: 0.8; cursor: pointer; }}
        .source-coverage:hover {{ opacity: 1; }}
        .event-marker {{ position: absolute; width: 2px; background: #e94560; cursor: pointer; }}
        .event-marker::before {{ content: ''; position: absolute; top: -6px; left: -4px; width: 10px; height: 10px; background: #e94560; border-radius: 50%; }}
        .event-tooltip {{ position: absolute; background: #0f3460; padding: 8px; border-radius: 4px; font-size: 12px; white-space: nowrap; z-index: 100; display: none; }}
        .event-marker:hover .event-tooltip {{ display: block; }}
        .utterance {{ padding: 10px; border-bottom: 1px solid #333; cursor: pointer; }}
        .utterance:hover {{ background: #0f3460; }}
        .utterance-time {{ font-size: 11px; color: #888; }}
        .utterance-speaker {{ font-weight: bold; color: #4285F4; }}
        .utterance-text {{ margin-top: 5px; }}
        .controls {{ display: flex; gap: 10px; margin-bottom: 10px; }}
        .controls button {{ padding: 8px 16px; background: #0f3460; border: none; color: #eee; border-radius: 4px; cursor: pointer; }}
        .controls button:hover {{ background: #1a4980; }}
        .playhead {{ position: absolute; width: 2px; background: #fff; z-index: 50; pointer-events: none; }}
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
            </div>
            <div class="timeline-container" id="timelineContainer">
                <div class="timeline" id="timeline">
                    <div class="time-axis" id="timeAxis"></div>
                    <div id="tracks" style="margin-top: 35px;">
                        {source_tracks}
                    </div>
                    <div id="events">
                        {event_markers}
                    </div>
                    <div class="playhead" id="playhead" style="display: none;"></div>
                </div>
            </div>
        </div>
        <div class="transcript-panel">
            <h2>Transcript</h2>
            <div id="transcript">
                {utterance_data}
            </div>
        </div>
    </div>
    <script>
        const DURATION_MS = {duration_ms};
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
            const width = document.getElementById('timeline').offsetWidth;
            const interval = Math.max(60000, Math.floor(DURATION_MS / (width / 100)));
            axis.innerHTML = '';
            for (let t = 0; t <= DURATION_MS; t += interval) {{
                const label = document.createElement('div');
                label.className = 'time-label';
                label.style.left = (t / DURATION_MS * 100) + '%';
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
            const playhead = document.getElementById('playhead');
            playhead.style.display = 'block';
            playhead.style.left = (ms / DURATION_MS * 100) + '%';
            playhead.style.height = document.getElementById('tracks').offsetHeight + 'px';
            playhead.style.top = '35px';

            // Scroll transcript to this time
            const utterances = document.querySelectorAll('.utterance');
            for (const u of utterances) {{
                const time = parseInt(u.dataset.time);
                if (time >= ms) {{
                    u.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    break;
                }}
            }}
        }}

        // Click handlers for coverage blocks
        document.querySelectorAll('.source-coverage').forEach(el => {{
            el.addEventListener('click', (e) => {{
                const rect = el.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const ratio = x / rect.width;
                const start = parseInt(el.dataset.start);
                const end = parseInt(el.dataset.end);
                const time = start + ratio * (end - start);
                seekTo(Math.floor(time));
            }});
        }});

        // Click handlers for utterances
        document.querySelectorAll('.utterance').forEach(el => {{
            el.addEventListener('click', () => {{
                seekTo(parseInt(el.dataset.time));
            }});
        }});

        updateTimeAxis();
    </script>
</body>
</html>"""

        return html

    def _generate_source_tracks(
        self,
        sources: list[Source],
        duration_ms: int,
    ) -> str:
        """Generate HTML for source tracks."""
        tracks = []

        for i, source in enumerate(sources):
            color = self.COLORS[i % len(self.COLORS)]
            label = f"{source.device_type.value} {source.device_number}"

            # Calculate coverage blocks
            coverage_blocks = []
            for file in source.files:
                start_ms = source.canonical_offset_ms + file.start_offset_ms
                end_ms = start_ms + int(file.duration_seconds * 1000)

                left_pct = (start_ms / duration_ms) * 100
                width_pct = ((end_ms - start_ms) / duration_ms) * 100

                coverage_blocks.append(f"""
                    <div class="source-coverage"
                         style="left: {left_pct}%; width: {width_pct}%; background: {color};"
                         data-start="{start_ms}" data-end="{end_ms}"
                         title="{file.filename}">
                    </div>
                """)

            tracks.append(f"""
                <div class="source-track">
                    <div class="source-label">{label}</div>
                    {''.join(coverage_blocks)}
                </div>
            """)

        return "\n".join(tracks)

    def _generate_event_markers(
        self,
        events: list[DeploymentEvent],
        duration_ms: int,
    ) -> str:
        """Generate HTML for event markers."""
        markers = []

        for event in events:
            left_pct = (event.canonical_time_ms / duration_ms) * 100

            markers.append(f"""
                <div class="event-marker"
                     style="left: {left_pct}%; height: 100px; top: 40px;">
                    <div class="event-tooltip">
                        <strong>{event.title}</strong><br>
                        {event.description or ''}
                    </div>
                </div>
            """)

        return "\n".join(markers)

    def _generate_utterance_data(
        self,
        utterances: list[CanonicalUtterance],
        people_names: dict[str, str],
    ) -> str:
        """Generate HTML for utterance list."""
        items = []

        for utterance in utterances[:500]:  # Limit for performance
            speaker = people_names.get(utterance.speaker_id, "Unknown")
            time_str = self._format_time(utterance.canonical_start_ms)

            items.append(f"""
                <div class="utterance" data-time="{utterance.canonical_start_ms}">
                    <div class="utterance-time">{time_str}</div>
                    <div class="utterance-speaker">{speaker}</div>
                    <div class="utterance-text">{utterance.text}</div>
                </div>
            """)

        return "\n".join(items)

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS."""
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def generate_gantt_chart(
        self,
        deployment: Deployment,
        events: list[DeploymentEvent],
    ) -> str:
        """Generate Gantt-style SVG chart of deployment phases."""
        # Filter to phase events
        phases = [e for e in events if e.event_type.value.startswith("phase_")]

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
        by_speaker: dict[str, list[CanonicalUtterance]] = {}
        for u in utterances:
            speaker = people_names.get(u.speaker_id, u.speaker_id or "Unknown")
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(u)

        # Calculate duration
        if utterances:
            duration_ms = max(u.canonical_end_ms for u in utterances)
        else:
            duration_ms = 3600000

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
        for i, (speaker, speaker_utterances) in enumerate(by_speaker.items()):
            color = self.COLORS[i % len(self.COLORS)]

            svg += f'<text x="10" y="{y + 25}" class="speaker-label">{speaker}</text>'

            for u in speaker_utterances:
                x = 150 + (u.canonical_start_ms / duration_ms) * (width - 160)
                w = max(2, (u.duration_ms / duration_ms) * (width - 160))
                svg += f'<rect x="{x}" y="{y + 5}" width="{w}" height="30" fill="{color}" class="utterance-block" rx="2"/>'

            y += row_height

        svg += "</svg>"
        return svg
