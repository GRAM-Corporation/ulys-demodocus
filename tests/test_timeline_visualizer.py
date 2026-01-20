"""Tests for the TimelineVisualizer service."""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.canonical_utterance import CanonicalUtterance, UtteranceSource
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.event import DeploymentEvent, EventType, Severity, ExtractionMethod
from gram_deploy.models.action_item import ActionItem, ActionItemStatus, Priority
from gram_deploy.models.source import Source, SourceFile, DeviceType, TranscriptStatus

# Import timeline_visualizer directly to avoid loading all services
import importlib.util

spec = importlib.util.spec_from_file_location(
    "timeline_visualizer",
    Path(__file__).parent.parent
    / "src"
    / "gram_deploy"
    / "services"
    / "timeline_visualizer.py",
)
tv_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv_module)
TimelineVisualizer = tv_module.TimelineVisualizer


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def service(temp_data_dir):
    """Create a TimelineVisualizer with test data directory."""
    return TimelineVisualizer(data_dir=temp_data_dir)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
        canonical_start_time=datetime(2025, 1, 19, 9, 0, 0, tzinfo=timezone.utc),
        canonical_end_time=datetime(2025, 1, 19, 12, 0, 0, tzinfo=timezone.utc),
        status=DeploymentStatus.COMPLETE,
    )


@pytest.fixture
def sample_utterances():
    """Create sample canonical utterances for testing."""
    return [
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/abc123",
            deployment_id="deploy:20250119_vinci_01",
            text="We need to check the battery levels before we start.",
            canonical_start_ms=0,
            canonical_end_ms=5000,
            speaker_id="person:damion",
            speaker_confidence=0.9,
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=0.0,
                    local_end_time=5.0,
                )
            ],
        ),
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/def456",
            deployment_id="deploy:20250119_vinci_01",
            text="I'll handle the Starlink setup. We should test connectivity first.",
            canonical_start_ms=6000,
            canonical_end_ms=12000,
            speaker_id="person:chu",
            speaker_confidence=0.85,
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=6.0,
                    local_end_time=12.0,
                )
            ],
        ),
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/ghi789",
            deployment_id="deploy:20250119_vinci_01",
            text="Let's document this process for the next deployment.",
            canonical_start_ms=15000,
            canonical_end_ms=20000,
            speaker_id="person:damion",
            speaker_confidence=0.9,
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=15.0,
                    local_end_time=20.0,
                )
            ],
        ),
    ]


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        DeploymentEvent(
            id="event:deploy:20250119_vinci_01/evt001",
            deployment_id="deploy:20250119_vinci_01",
            event_type=EventType.DECISION,
            canonical_time_ms=6000,
            title="Decided to use Starlink",
            description="Team decided to set up Starlink for connectivity",
            severity=Severity.INFO,
            extraction_method=ExtractionMethod.LLM_EXTRACTED,
            confidence=0.85,
        ),
        DeploymentEvent(
            id="event:deploy:20250119_vinci_01/evt002",
            deployment_id="deploy:20250119_vinci_01",
            event_type=EventType.ISSUE,
            canonical_time_ms=30000,
            title="Battery low warning",
            description="Battery level dropped below 20%",
            severity=Severity.WARNING,
            extraction_method=ExtractionMethod.LLM_EXTRACTED,
            confidence=0.9,
        ),
    ]


@pytest.fixture
def sample_action_items():
    """Create sample action items for testing."""
    return [
        ActionItem(
            id="action:deploy:20250119_vinci_01/act001",
            deployment_id="deploy:20250119_vinci_01",
            description="Document the deployment process",
            source_utterance_id="utterance:deploy:20250119_vinci_01/ghi789",
            canonical_time_ms=15000,
            mentioned_by="Damion Shelton",
            priority=Priority.MEDIUM,
            status=ActionItemStatus.EXTRACTED,
            extraction_confidence=0.9,
        ),
        ActionItem(
            id="action:deploy:20250119_vinci_01/act002",
            deployment_id="deploy:20250119_vinci_01",
            description="Order backup batteries",
            source_utterance_id="utterance:deploy:20250119_vinci_01/abc123",
            canonical_time_ms=5000,
            mentioned_by="Chu",
            priority=Priority.HIGH,
            status=ActionItemStatus.EXTRACTED,
            extraction_confidence=0.8,
        ),
    ]


@pytest.fixture
def sample_sources():
    """Create sample sources for testing."""
    return [
        Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            files=[
                SourceFile(
                    filename="GX010001.MP4",
                    file_path="/path/to/GX010001.MP4",
                    duration_seconds=1800.0,
                    start_offset_ms=0,
                ),
                SourceFile(
                    filename="GX010002.MP4",
                    file_path="/path/to/GX010002.MP4",
                    duration_seconds=1800.0,
                    start_offset_ms=1800000,
                ),
            ],
            total_duration_seconds=3600.0,
            canonical_offset_ms=0,
            alignment_confidence=0.95,
            transcript_status=TranscriptStatus.COMPLETE,
        ),
    ]


@pytest.fixture
def people_names():
    """Create sample people names mapping."""
    return {
        "person:damion": "Damion Shelton",
        "person:chu": "Chu",
    }


class TestDataPreparation:
    """Tests for data preparation methods."""

    def test_group_by_speaker(self, service, sample_utterances, people_names):
        """Test grouping utterances by speaker."""
        result = service._group_by_speaker(sample_utterances, people_names)

        assert "Damion Shelton" in result
        assert "Chu" in result
        assert len(result["Damion Shelton"]) == 2
        assert len(result["Chu"]) == 1

    def test_group_by_speaker_unknown(self, service, sample_utterances):
        """Test grouping with unknown speakers."""
        result = service._group_by_speaker(sample_utterances, {})

        # Speaker IDs should be used as fallback
        assert "person:damion" in result or "Unknown" in result

    def test_assign_speaker_colors(self, service):
        """Test color assignment to speakers."""
        speakers = ["Alice", "Bob", "Charlie"]
        colors = service._assign_speaker_colors(speakers)

        assert len(colors) == 3
        assert all(color.startswith("#") for color in colors.values())
        # Colors should be consistent (sorted speakers)
        assert colors["Alice"] == service.SPEAKER_COLORS[0]

    def test_prepare_utterance_data(self, service, sample_utterances, people_names):
        """Test preparing utterance data for JSON."""
        result = service._prepare_utterance_data(sample_utterances, people_names)

        assert len(result) == 3
        assert result[0]["text"] == "We need to check the battery levels before we start."
        assert result[0]["speaker_name"] == "Damion Shelton"
        assert result[0]["start_ms"] == 0
        assert result[0]["end_ms"] == 5000
        assert "speaker_color" in result[0]

    def test_prepare_event_data(self, service, sample_events):
        """Test preparing event data for JSON."""
        result = service._prepare_event_data(sample_events, 3600000)

        assert len(result) == 2
        assert result[0]["title"] == "Decided to use Starlink"
        assert result[0]["type"] == "decision"
        assert result[0]["icon"] == service.EVENT_ICONS[EventType.DECISION]
        assert result[0]["color"] == service.EVENT_COLORS[EventType.DECISION]

    def test_prepare_action_item_data(self, service, sample_action_items):
        """Test preparing action item data for JSON."""
        result = service._prepare_action_item_data(sample_action_items, 3600000)

        assert len(result) == 2
        assert result[0]["description"] == "Document the deployment process"
        assert result[0]["priority"] == "medium"
        assert result[0]["priority_color"] == service.PRIORITY_COLORS[Priority.MEDIUM]

    def test_prepare_speaker_lane_data(self, service, sample_utterances, people_names):
        """Test preparing speaker lane data for JSON."""
        speaker_lanes = service._group_by_speaker(sample_utterances, people_names)
        speaker_colors = service._assign_speaker_colors(speaker_lanes.keys())
        result = service._prepare_speaker_lane_data(speaker_lanes, speaker_colors, 3600000)

        assert len(result) == 2
        # Should be sorted alphabetically
        assert result[0]["speaker_name"] == "Chu"
        assert result[1]["speaker_name"] == "Damion Shelton"
        assert len(result[1]["utterances"]) == 2

    def test_prepare_source_track_data(self, service, sample_sources):
        """Test preparing source track data for JSON."""
        result = service._prepare_source_track_data(sample_sources, 3600000)

        assert len(result) == 1
        assert result[0]["label"] == "gopro 1"
        assert len(result[0]["segments"]) == 2
        assert result[0]["segments"][0]["filename"] == "GX010001.MP4"

    def test_calculate_duration_from_deployment(self, service, sample_deployment):
        """Test duration calculation from deployment times."""
        duration = service._calculate_duration(sample_deployment, [], [])

        # 3 hours = 10,800,000 ms
        assert duration == 10800000

    def test_calculate_duration_from_utterances(self, service, sample_utterances):
        """Test duration calculation from utterances."""
        deployment = Deployment(
            id="deploy:20250119_test_01",
            location="test",
            date="2025-01-19",
            status=DeploymentStatus.COMPLETE,
        )
        duration = service._calculate_duration(deployment, sample_utterances, [])

        # Last utterance ends at 20000ms
        assert duration == 20000

    def test_calculate_duration_default(self, service):
        """Test default duration when no data available."""
        deployment = Deployment(
            id="deploy:20250119_test_01",
            location="test",
            date="2025-01-19",
            status=DeploymentStatus.COMPLETE,
        )
        duration = service._calculate_duration(deployment, [], [])

        # Default 1 hour
        assert duration == 3600000


class TestHtmlGeneration:
    """Tests for HTML generation methods."""

    def test_generate_speaker_lanes_html(self, service, sample_utterances, people_names):
        """Test speaker lanes HTML generation."""
        speaker_lanes = service._group_by_speaker(sample_utterances, people_names)
        speaker_colors = service._assign_speaker_colors(speaker_lanes.keys())
        lane_data = service._prepare_speaker_lane_data(speaker_lanes, speaker_colors, 3600000)
        html = service._generate_speaker_lanes_html(lane_data, 3600000)

        assert "speaker-lane" in html
        assert "Damion Shelton" in html
        assert "Chu" in html
        assert "utterance-block" in html

    def test_generate_speaker_lanes_html_empty(self, service):
        """Test speaker lanes HTML with no data."""
        html = service._generate_speaker_lanes_html([], 3600000)

        assert "No speaker data available" in html

    def test_generate_event_markers_html(self, service, sample_events):
        """Test event markers HTML generation."""
        event_data = service._prepare_event_data(sample_events, 3600000)
        html = service._generate_event_markers_html(event_data, 3600000)

        assert "event-marker" in html
        assert "Decided to use Starlink" in html
        assert "event-tooltip" in html

    def test_generate_event_markers_html_empty(self, service):
        """Test event markers HTML with no data."""
        html = service._generate_event_markers_html([], 3600000)

        assert "No events recorded" in html

    def test_generate_action_items_html(self, service, sample_action_items):
        """Test action items HTML generation."""
        item_data = service._prepare_action_item_data(sample_action_items, 3600000)
        html = service._generate_action_items_html(item_data, 3600000)

        assert "action-marker" in html
        assert "Document the deployment process" in html
        assert "priority-badge" in html

    def test_generate_action_items_html_empty(self, service):
        """Test action items HTML with no data."""
        html = service._generate_action_items_html([], 3600000)

        assert "No action items recorded" in html

    def test_generate_source_tracks_html(self, service, sample_sources):
        """Test source tracks HTML generation."""
        track_data = service._prepare_source_track_data(sample_sources, 3600000)
        html = service._generate_source_tracks_html(track_data, 3600000)

        assert "source-track" in html
        assert "gopro 1" in html
        assert "source-coverage" in html

    def test_generate_utterances_html(self, service, sample_utterances, people_names):
        """Test utterances HTML generation."""
        utterance_data = service._prepare_utterance_data(sample_utterances, people_names)
        html = service._generate_utterances_html(utterance_data)

        assert "utterance" in html
        assert "We need to check the battery levels" in html
        assert "Damion Shelton" in html


class TestHtmlOutput:
    """Tests for complete HTML output structure."""

    def test_generate_timeline_html_structure(
        self, service, sample_deployment, sample_sources,
        sample_utterances, sample_events, people_names, sample_action_items
    ):
        """Test that generated HTML has correct structure."""
        html = service.generate_timeline_html(
            sample_deployment,
            sample_sources,
            sample_utterances,
            sample_events,
            people_names,
            sample_action_items,
        )

        # Check document structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html

        # Check title
        assert "vinci" in html
        assert "2025-01-19" in html

        # Check main sections
        assert "timeline-panel" in html
        assert "transcript-panel" in html
        assert "controls" in html

        # Check speaker section
        assert "speaker-section" in html
        assert "Speaker Activity" in html

        # Check events section
        assert "events-section" in html
        assert "Events" in html

        # Check action items section
        assert "action-items-section" in html
        assert "Action Items" in html

        # Check sources section
        assert "sources-section" in html
        assert "Source Coverage" in html

        # Check JavaScript functions
        assert "zoomIn()" in html
        assert "zoomOut()" in html
        assert "resetZoom()" in html
        assert "seekTo(" in html

    def test_generate_timeline_html_contains_data(
        self, service, sample_deployment, sample_sources,
        sample_utterances, sample_events, people_names
    ):
        """Test that generated HTML contains actual data."""
        html = service.generate_timeline_html(
            sample_deployment,
            sample_sources,
            sample_utterances,
            sample_events,
            people_names,
        )

        # Check utterance data
        assert "battery levels" in html
        assert "Starlink" in html

        # Check speaker names
        assert "Damion Shelton" in html
        assert "Chu" in html

        # Check event data
        assert "Decided to use Starlink" in html

    def test_generate_timeline_html_legend(
        self, service, sample_deployment, sample_sources,
        sample_utterances, sample_events, people_names
    ):
        """Test that legend is present."""
        html = service.generate_timeline_html(
            sample_deployment,
            sample_sources,
            sample_utterances,
            sample_events,
            people_names,
        )

        assert "legend" in html
        assert "Decision" in html
        assert "Issue" in html
        assert "Observation" in html
        assert "Milestone" in html


class TestFileIO:
    """Tests for file I/O operations."""

    def test_generate_timeline_creates_file(
        self, service, sample_deployment, temp_data_dir
    ):
        """Test that generate_timeline creates timeline.html file."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())

        # Write empty transcript
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        # Generate timeline
        output_path = service.generate_timeline(sample_deployment)

        assert output_path.exists()
        assert output_path.name == "timeline.html"

        # Check content
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_generate_timeline_with_deployment_id(
        self, service, sample_deployment, temp_data_dir
    ):
        """Test generate_timeline with deployment ID string."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())

        # Write empty transcript
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        # Generate timeline with ID string
        output_path = service.generate_timeline("deploy:20250119_vinci_01")

        assert output_path.exists()

    def test_generate_timeline_loads_all_data(
        self, service, sample_deployment, sample_utterances,
        sample_events, sample_action_items, people_names, temp_data_dir
    ):
        """Test that generate_timeline loads all data types."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())

        # Write transcript
        transcript_data = [u.model_dump(mode="json") for u in sample_utterances]
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str)
        )

        # Write events
        events_data = [e.model_dump(mode="json") for e in sample_events]
        (deploy_dir / "events.json").write_text(
            json.dumps(events_data, indent=2, default=str)
        )

        # Write action items
        items_data = [i.model_dump(mode="json") for i in sample_action_items]
        (deploy_dir / "action_items.json").write_text(
            json.dumps(items_data, indent=2, default=str)
        )

        # Write people registry
        people_data = {
            "people": [
                {"id": "person:damion", "name": "Damion Shelton"},
                {"id": "person:chu", "name": "Chu"},
            ]
        }
        (Path(temp_data_dir) / "people.json").write_text(json.dumps(people_data))

        # Generate timeline
        output_path = service.generate_timeline(sample_deployment)

        # Check that all data is present in output
        content = output_path.read_text()
        assert "Damion Shelton" in content
        assert "Decided to use Starlink" in content
        assert "Document the deployment process" in content

    def test_generate_timeline_deployment_not_found(self, service, temp_data_dir):
        """Test that generate_timeline raises error for missing deployment."""
        with pytest.raises(FileNotFoundError):
            service.generate_timeline("deploy:20250119_nonexistent_01")

    def test_load_canonical_utterances_array_format(
        self, service, sample_utterances, temp_data_dir
    ):
        """Test loading utterances from array format JSON."""
        deploy_dir = Path(temp_data_dir) / "test_deploy"
        (deploy_dir / "canonical").mkdir(parents=True)

        # Write in array format
        transcript_data = [u.model_dump(mode="json") for u in sample_utterances]
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str)
        )

        utterances = service._load_canonical_utterances(deploy_dir)

        assert len(utterances) == 3

    def test_load_canonical_utterances_object_format(
        self, service, sample_utterances, temp_data_dir
    ):
        """Test loading utterances from object format JSON."""
        deploy_dir = Path(temp_data_dir) / "test_deploy"
        (deploy_dir / "canonical").mkdir(parents=True)

        # Write in object format
        transcript_data = {
            "utterances": [u.model_dump(mode="json") for u in sample_utterances]
        }
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str)
        )

        utterances = service._load_canonical_utterances(deploy_dir)

        assert len(utterances) == 3

    def test_load_events(self, service, sample_events, temp_data_dir):
        """Test loading events from deployment directory."""
        deploy_dir = Path(temp_data_dir) / "test_deploy"
        deploy_dir.mkdir(parents=True)

        events_data = [e.model_dump(mode="json") for e in sample_events]
        (deploy_dir / "events.json").write_text(
            json.dumps(events_data, indent=2, default=str)
        )

        events = service._load_events(deploy_dir)

        assert len(events) == 2
        assert events[0].title == "Decided to use Starlink"

    def test_load_action_items(self, service, sample_action_items, temp_data_dir):
        """Test loading action items from deployment directory."""
        deploy_dir = Path(temp_data_dir) / "test_deploy"
        deploy_dir.mkdir(parents=True)

        items_data = [i.model_dump(mode="json") for i in sample_action_items]
        (deploy_dir / "action_items.json").write_text(
            json.dumps(items_data, indent=2, default=str)
        )

        items = service._load_action_items(deploy_dir)

        assert len(items) == 2
        assert items[0].description == "Document the deployment process"

    def test_load_people_names(self, service, temp_data_dir):
        """Test loading people names from registry."""
        people_data = {
            "people": [
                {"id": "person:alice", "name": "Alice Smith"},
                {"id": "person:bob", "name": "Bob Jones"},
            ]
        }
        (Path(temp_data_dir) / "people.json").write_text(json.dumps(people_data))

        names = service._load_people_names(Path(temp_data_dir))

        assert names == {
            "person:alice": "Alice Smith",
            "person:bob": "Bob Jones",
        }


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_format_time(self, service):
        """Test time formatting."""
        assert service._format_time(0) == "00:00:00"
        assert service._format_time(1000) == "00:00:01"
        assert service._format_time(60000) == "00:01:00"
        assert service._format_time(3600000) == "01:00:00"
        assert service._format_time(3661000) == "01:01:01"

    def test_escape_html(self, service):
        """Test HTML escaping."""
        assert service._escape_html("<script>") == "&lt;script&gt;"
        assert service._escape_html('"quoted"') == "&quot;quoted&quot;"
        assert service._escape_html("'single'") == "&#39;single&#39;"
        assert service._escape_html("a & b") == "a &amp; b"

    def test_get_deployment_dir(self, service, temp_data_dir):
        """Test deployment directory path calculation."""
        path = service._get_deployment_dir(
            "deploy:20250119_vinci_01",
            Path(temp_data_dir)
        )

        assert path == Path(temp_data_dir) / "deploy_20250119_vinci_01"


class TestLegacyMethods:
    """Tests for backward compatibility methods."""

    def test_generate_gantt_chart(self, service, sample_deployment, sample_events):
        """Test Gantt chart generation."""
        # Add a phase event
        phase_event = DeploymentEvent(
            id="event:deploy:20250119_vinci_01/phase001",
            deployment_id="deploy:20250119_vinci_01",
            event_type=EventType.PHASE_START,
            canonical_time_ms=0,
            title="Setup Phase",
            extraction_method=ExtractionMethod.MANUAL,
        )
        events = [phase_event] + sample_events

        svg = service.generate_gantt_chart(sample_deployment, events)

        assert "<svg" in svg
        assert "Setup Phase" in svg

    def test_generate_speaker_timeline(
        self, service, sample_deployment, sample_utterances, people_names
    ):
        """Test speaker timeline SVG generation."""
        svg = service.generate_speaker_timeline(
            sample_deployment, sample_utterances, people_names
        )

        assert "<svg" in svg
        assert "Damion Shelton" in svg
        assert "Chu" in svg
