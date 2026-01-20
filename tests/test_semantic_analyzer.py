"""Tests for the SemanticAnalyzer service."""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.canonical_utterance import CanonicalUtterance, UtteranceSource
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.event import DeploymentEvent, EventType, Severity, ExtractionMethod
from gram_deploy.models.action_item import ActionItem, ActionItemStatus, Priority
from gram_deploy.models.insight import DeploymentInsight, InsightType, SupportingEvidence

# Import semantic_analyzer directly to avoid loading all services
import importlib.util

spec = importlib.util.spec_from_file_location(
    "semantic_analyzer",
    Path(__file__).parent.parent
    / "src"
    / "gram_deploy"
    / "services"
    / "semantic_analyzer.py",
)
sa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa_module)
SemanticAnalyzer = sa_module.SemanticAnalyzer
SemanticAnalysisResult = sa_module.SemanticAnalysisResult


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    return client


@pytest.fixture
def service(mock_llm_client, temp_data_dir):
    """Create a SemanticAnalyzer with mock LLM client."""
    cache_dir = Path(temp_data_dir) / "cache"
    return SemanticAnalyzer(mock_llm_client, cache_dir=str(cache_dir), data_dir=temp_data_dir)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
        canonical_start_time=datetime(2025, 1, 19, 9, 0, 0, tzinfo=timezone.utc),
        canonical_end_time=datetime(2025, 1, 19, 12, 0, 0, tzinfo=timezone.utc),
        status=DeploymentStatus.ANALYZING,
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
def people_names():
    """Create sample people names mapping."""
    return {
        "person:damion": "Damion Shelton",
        "person:chu": "Chu",
    }


class TestServiceInitialization:
    """Tests for service initialization."""

    def test_init_creates_cache_dir(self, mock_llm_client, temp_data_dir):
        """Test that service creates cache directory."""
        cache_dir = Path(temp_data_dir) / "test_cache"
        service = SemanticAnalyzer(mock_llm_client, cache_dir=str(cache_dir))

        assert cache_dir.exists()

    def test_init_with_default_cache_dir(self, mock_llm_client, temp_data_dir):
        """Test that service uses default cache dir when not specified."""
        service = SemanticAnalyzer(mock_llm_client, data_dir=temp_data_dir)

        expected_cache = Path(temp_data_dir) / "cache" / "llm_responses"
        assert service.cache_dir == expected_cache


class TestPromptFormatting:
    """Tests for LLM prompt formatting."""

    def test_build_transcript_text(self, service, sample_utterances, people_names):
        """Test that transcript text is formatted correctly."""
        transcript_text = service._build_transcript_text(sample_utterances, people_names)

        assert "[00:00:00] Damion Shelton:" in transcript_text
        assert "[00:00:06] Chu:" in transcript_text
        assert "battery levels" in transcript_text
        assert "Starlink setup" in transcript_text

    def test_build_transcript_text_unknown_speaker(self, service, sample_utterances):
        """Test transcript formatting with unknown speaker."""
        transcript_text = service._build_transcript_text(sample_utterances, {})

        assert "[00:00:00] Unknown:" in transcript_text

    def test_format_time(self, service):
        """Test time formatting helper."""
        assert service._format_time(0) == "00:00:00"
        assert service._format_time(1000) == "00:00:01"
        assert service._format_time(60000) == "00:01:00"
        assert service._format_time(3600000) == "01:00:00"
        assert service._format_time(3661000) == "01:01:01"

    def test_get_speaker_names(self, service, sample_utterances, people_names):
        """Test getting unique speaker names."""
        speakers = service._get_speaker_names(sample_utterances, people_names)

        assert "Damion Shelton" in speakers
        assert "Chu" in speakers
        assert len(speakers) == 2


class TestSegmentation:
    """Tests for transcript segmentation."""

    def test_segment_utterances_single_segment(self, service, sample_utterances):
        """Test that short transcript stays in single segment."""
        segments = service._segment_utterances(sample_utterances)

        assert len(segments) == 1
        assert len(segments[0][1]) == 3

    def test_segment_utterances_multiple_segments(self, service):
        """Test that long transcript is split into segments."""
        # Create utterances spanning 25 minutes
        utterances = []
        for i in range(25):
            utterances.append(
                CanonicalUtterance(
                    id=f"utterance:deploy:20250119_vinci_01/test{i:03d}",
                    deployment_id="deploy:20250119_vinci_01",
                    text=f"Utterance at minute {i}",
                    canonical_start_ms=i * 60 * 1000,
                    canonical_end_ms=(i * 60 + 30) * 1000,
                    speaker_id="person:damion",
                    speaker_confidence=0.9,
                )
            )

        segments = service._segment_utterances(utterances)

        # 25 minutes should be split into 3 segments (10 + 10 + 5)
        assert len(segments) >= 2

    def test_segment_utterances_empty(self, service):
        """Test segmenting empty utterances list."""
        segments = service._segment_utterances([])
        assert segments == []


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_extract_json_valid(self, service):
        """Test extracting JSON from LLM response."""
        response = 'Here are the results:\n[{"key": "value"}]\nDone.'
        result = service._extract_json(response)

        assert result == [{"key": "value"}]

    def test_extract_json_no_json(self, service):
        """Test extracting JSON when none present."""
        response = "No JSON here."
        result = service._extract_json(response)

        assert result == []

    def test_parse_events_response(self, service):
        """Test parsing events from LLM response."""
        response = json.dumps([
            {
                "event_type": "decision",
                "title": "Decided to use Starlink",
                "description": "Team decided to set up Starlink for connectivity",
                "time_offset_ms": 5000,
                "severity": "info",
                "confidence": 0.85,
            }
        ])

        events = service._parse_events_response(
            response, "deploy:20250119_vinci_01", 0
        )

        assert len(events) == 1
        assert events[0].event_type == EventType.DECISION
        assert events[0].title == "Decided to use Starlink"
        assert events[0].canonical_time_ms == 5000
        assert events[0].severity == Severity.INFO
        assert events[0].confidence == 0.85
        assert events[0].extraction_method == ExtractionMethod.LLM_EXTRACTED

    def test_parse_action_items_response(self, service, sample_utterances):
        """Test parsing action items from LLM response."""
        response = json.dumps([
            {
                "description": "Document the deployment process",
                "mentioned_by": "Damion Shelton",
                "assigned_to": None,
                "priority": "medium",
                "time_offset_ms": 15000,
                "confidence": 0.9,
            }
        ])

        action_items = service._parse_action_items_response(
            response, "deploy:20250119_vinci_01", 0, sample_utterances
        )

        assert len(action_items) == 1
        assert action_items[0].description == "Document the deployment process"
        assert action_items[0].mentioned_by == "Damion Shelton"
        assert action_items[0].priority == Priority.MEDIUM
        assert action_items[0].status == ActionItemStatus.EXTRACTED

    def test_parse_insights_response(self, service):
        """Test parsing insights from LLM response."""
        response = json.dumps([
            {
                "insight_type": "process_improvement",
                "content": "Pre-deployment battery checks improve efficiency",
                "supporting_quote": "We need to check the battery levels before we start",
                "confidence": 0.8,
            }
        ])

        insights = service._parse_insights_response(
            response, "deploy:20250119_vinci_01"
        )

        assert len(insights) == 1
        assert insights[0].insight_type == InsightType.PROCESS_IMPROVEMENT
        assert "battery checks" in insights[0].content
        assert len(insights[0].supporting_evidence) == 1

    def test_parse_events_response_invalid_json(self, service):
        """Test that invalid JSON returns empty list."""
        events = service._parse_events_response(
            "not valid json", "deploy:20250119_vinci_01", 0
        )
        assert events == []


class TestDeduplication:
    """Tests for result deduplication."""

    def test_deduplicate_events(self, service):
        """Test deduplicating similar events."""
        events = [
            DeploymentEvent(
                id="event:deploy:20250119_vinci_01/abc123",
                deployment_id="deploy:20250119_vinci_01",
                event_type=EventType.DECISION,
                canonical_time_ms=5000,
                title="Decided to use Starlink",
                extraction_method=ExtractionMethod.LLM_EXTRACTED,
            ),
            DeploymentEvent(
                id="event:deploy:20250119_vinci_01/def456",
                deployment_id="deploy:20250119_vinci_01",
                event_type=EventType.DECISION,
                canonical_time_ms=5500,  # Same minute
                title="Decided to use Starlink",
                extraction_method=ExtractionMethod.LLM_EXTRACTED,
            ),
        ]

        deduplicated = service._deduplicate_events(events)

        assert len(deduplicated) == 1

    def test_deduplicate_action_items(self, service):
        """Test deduplicating similar action items."""
        # Use descriptions that match in the first 50 characters (after lowercasing)
        items = [
            ActionItem(
                id="action:deploy:20250119_vinci_01/abc123",
                deployment_id="deploy:20250119_vinci_01",
                description="Document the deployment process",
                source_utterance_id="utterance:test",
                canonical_time_ms=5000,
            ),
            ActionItem(
                id="action:deploy:20250119_vinci_01/def456",
                deployment_id="deploy:20250119_vinci_01",
                description="Document the deployment process",  # Same description
                source_utterance_id="utterance:test",
                canonical_time_ms=6000,
            ),
        ]

        deduplicated = service._deduplicate_action_items(items)

        # Identical descriptions (first 50 chars match) should deduplicate
        assert len(deduplicated) == 1

    def test_deduplicate_action_items_different(self, service):
        """Test that different action items are not deduplicated."""
        items = [
            ActionItem(
                id="action:deploy:20250119_vinci_01/abc123",
                deployment_id="deploy:20250119_vinci_01",
                description="Document the deployment process",
                source_utterance_id="utterance:test",
                canonical_time_ms=5000,
            ),
            ActionItem(
                id="action:deploy:20250119_vinci_01/def456",
                deployment_id="deploy:20250119_vinci_01",
                description="Check battery levels before next deployment",
                source_utterance_id="utterance:test",
                canonical_time_ms=6000,
            ),
        ]

        deduplicated = service._deduplicate_action_items(items)

        # Different descriptions should not deduplicate
        assert len(deduplicated) == 2

    def test_deduplicate_insights(self, service):
        """Test deduplicating similar insights."""
        # Use content that matches in the first 50 characters
        insights = [
            DeploymentInsight(
                id="insight:deploy:20250119_vinci_01/abc123",
                deployment_id="deploy:20250119_vinci_01",
                insight_type=InsightType.TECHNICAL_OBSERVATION,
                content="Battery checks are important for deployment",
            ),
            DeploymentInsight(
                id="insight:deploy:20250119_vinci_01/def456",
                deployment_id="deploy:20250119_vinci_01",
                insight_type=InsightType.TECHNICAL_OBSERVATION,
                content="Battery checks are important for deployment",  # Same content
            ),
        ]

        deduplicated = service._deduplicate_insights(insights)

        assert len(deduplicated) == 1

    def test_deduplicate_insights_different(self, service):
        """Test that different insights are not deduplicated."""
        insights = [
            DeploymentInsight(
                id="insight:deploy:20250119_vinci_01/abc123",
                deployment_id="deploy:20250119_vinci_01",
                insight_type=InsightType.TECHNICAL_OBSERVATION,
                content="Battery checks are important for deployment success",
            ),
            DeploymentInsight(
                id="insight:deploy:20250119_vinci_01/def456",
                deployment_id="deploy:20250119_vinci_01",
                insight_type=InsightType.TECHNICAL_OBSERVATION,
                content="Starlink setup should be verified before field work",
            ),
        ]

        deduplicated = service._deduplicate_insights(insights)

        # Different content should not deduplicate
        assert len(deduplicated) == 2


class TestFindClosestUtterance:
    """Tests for finding closest utterance to a time."""

    def test_find_closest_utterance(self, service, sample_utterances):
        """Test finding utterance closest to given time."""
        closest = service._find_closest_utterance(sample_utterances, 7000)

        assert closest.id == "utterance:deploy:20250119_vinci_01/def456"

    def test_find_closest_utterance_empty(self, service):
        """Test finding closest utterance with empty list."""
        closest = service._find_closest_utterance([], 5000)
        assert closest is None


class TestAnalyzeDeployment:
    """Tests for the main analyze_deployment method."""

    def test_analyze_deployment_with_mock_llm(
        self, service, mock_llm_client, sample_deployment, sample_utterances, people_names
    ):
        """Test full analysis pipeline with mocked LLM."""
        # Setup mock responses
        events_response = json.dumps([
            {
                "event_type": "decision",
                "title": "Setup Starlink connectivity",
                "description": "Team decided to prioritize Starlink setup",
                "time_offset_ms": 6000,
                "severity": "info",
                "confidence": 0.8,
            }
        ])
        action_items_response = json.dumps([
            {
                "description": "Document deployment process",
                "mentioned_by": "Damion Shelton",
                "priority": "medium",
                "time_offset_ms": 15000,
                "confidence": 0.85,
            }
        ])
        insights_response = json.dumps([
            {
                "insight_type": "process_improvement",
                "content": "Pre-flight battery checks improve efficiency",
                "supporting_quote": "check the battery levels",
                "confidence": 0.75,
            }
        ])
        summary_response = "This deployment at vinci focused on setting up connectivity."

        # Mock LLM to return different responses for different prompts
        call_count = [0]
        responses = [events_response, action_items_response, insights_response, summary_response]

        def mock_create(**kwargs):
            response = MagicMock()
            response.content = [MagicMock(text=responses[min(call_count[0], len(responses) - 1)])]
            call_count[0] += 1
            return response

        mock_llm_client.messages.create = mock_create

        result = service.analyze_deployment(sample_deployment, sample_utterances, people_names)

        assert isinstance(result, SemanticAnalysisResult)
        assert len(result.events) >= 1
        assert len(result.action_items) >= 1
        assert len(result.insights) >= 1
        assert result.summary is not None

    def test_analyze_deployment_empty_utterances(
        self, service, sample_deployment
    ):
        """Test analysis with empty utterances returns empty result."""
        result = service.analyze_deployment(sample_deployment, [], {})

        assert result.events == []
        assert result.action_items == []
        assert result.insights == []


class TestAnalyzeWithFileIO:
    """Tests for the analyze method with file I/O."""

    def test_analyze_loads_and_saves(
        self, service, mock_llm_client, sample_deployment, sample_utterances, temp_data_dir
    ):
        """Test that analyze method loads transcripts and saves results."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())

        # Write transcript file
        transcript_data = [u.model_dump(mode="json") for u in sample_utterances]
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str)
        )

        # Write people registry
        people_data = {
            "people": [
                {"id": "person:damion", "name": "Damion Shelton"},
                {"id": "person:chu", "name": "Chu"},
            ]
        }
        (Path(temp_data_dir) / "people.json").write_text(json.dumps(people_data))

        # Setup mock responses
        events_response = json.dumps([
            {
                "event_type": "milestone",
                "title": "Deployment started",
                "time_offset_ms": 0,
                "severity": "info",
                "confidence": 0.9,
            }
        ])
        action_items_response = json.dumps([])
        insights_response = json.dumps([])
        summary_response = "Deployment summary."

        call_count = [0]
        responses = [events_response, action_items_response, insights_response, summary_response]

        def mock_create(**kwargs):
            response = MagicMock()
            response.content = [MagicMock(text=responses[min(call_count[0], len(responses) - 1)])]
            call_count[0] += 1
            return response

        mock_llm_client.messages.create = mock_create

        # Run analyze
        result = service.analyze(sample_deployment)

        # Check files were created
        assert (deploy_dir / "events.json").exists()
        assert (deploy_dir / "action_items.json").exists()
        assert (deploy_dir / "insights.json").exists()
        assert (deploy_dir / "summary.md").exists()

        # Check content
        events_data = json.loads((deploy_dir / "events.json").read_text())
        assert len(events_data) >= 1
        assert events_data[0]["title"] == "Deployment started"

    def test_analyze_with_deployment_id_string(
        self, service, mock_llm_client, sample_deployment, sample_utterances, temp_data_dir
    ):
        """Test that analyze method accepts deployment ID string."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())

        # Write transcript file (empty)
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        # Setup mock response
        mock_llm_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="[]")]
        )

        # Run analyze with ID string
        result = service.analyze("deploy:20250119_vinci_01")

        assert isinstance(result, SemanticAnalysisResult)

    def test_analyze_deployment_not_found(self, service, temp_data_dir):
        """Test that analyze raises error for missing deployment."""
        with pytest.raises(FileNotFoundError):
            service.analyze("deploy:20250119_nonexistent_01")


class TestCaching:
    """Tests for LLM response caching."""

    def test_llm_response_cached(self, service, mock_llm_client):
        """Test that LLM responses are cached."""
        mock_llm_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="cached response")]
        )

        # First call
        result1 = service._call_llm("test prompt")

        # Second call with same prompt
        result2 = service._call_llm("test prompt")

        # LLM should only be called once
        assert mock_llm_client.messages.create.call_count == 1
        assert result1 == result2

    def test_different_prompts_not_cached(self, service, mock_llm_client):
        """Test that different prompts are not served from cache."""
        mock_llm_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="response")]
        )

        service._call_llm("prompt 1")
        service._call_llm("prompt 2")

        assert mock_llm_client.messages.create.call_count == 2


class TestSaveResults:
    """Tests for saving analysis results."""

    def test_save_events(self, service, temp_data_dir):
        """Test saving events to file."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        events = [
            DeploymentEvent(
                id="event:deploy:20250119_vinci_01/test123",
                deployment_id="deploy:20250119_vinci_01",
                event_type=EventType.MILESTONE,
                canonical_time_ms=5000,
                title="Test event",
                extraction_method=ExtractionMethod.LLM_EXTRACTED,
            )
        ]

        service._save_events(events, deploy_dir)

        events_path = deploy_dir / "events.json"
        assert events_path.exists()

        saved_data = json.loads(events_path.read_text())
        assert len(saved_data) == 1
        assert saved_data[0]["title"] == "Test event"

    def test_save_action_items(self, service, temp_data_dir):
        """Test saving action items to file."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        items = [
            ActionItem(
                id="action:deploy:20250119_vinci_01/test123",
                deployment_id="deploy:20250119_vinci_01",
                description="Test action item",
                source_utterance_id="utterance:test",
                canonical_time_ms=5000,
            )
        ]

        service._save_action_items(items, deploy_dir)

        items_path = deploy_dir / "action_items.json"
        assert items_path.exists()

        saved_data = json.loads(items_path.read_text())
        assert len(saved_data) == 1
        assert saved_data[0]["description"] == "Test action item"

    def test_save_insights(self, service, temp_data_dir):
        """Test saving insights to file."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        insights = [
            DeploymentInsight(
                id="insight:deploy:20250119_vinci_01/test123",
                deployment_id="deploy:20250119_vinci_01",
                insight_type=InsightType.TECHNICAL_OBSERVATION,
                content="Test insight",
            )
        ]

        service._save_insights(insights, deploy_dir)

        insights_path = deploy_dir / "insights.json"
        assert insights_path.exists()

        saved_data = json.loads(insights_path.read_text())
        assert len(saved_data) == 1
        assert saved_data[0]["content"] == "Test insight"

    def test_save_summary(self, service, temp_data_dir):
        """Test saving summary to file."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        summary = "This is the deployment summary."

        service._save_summary(summary, deploy_dir)

        summary_path = deploy_dir / "summary.md"
        assert summary_path.exists()
        assert summary_path.read_text() == summary


class TestLoadCanonicalUtterances:
    """Tests for loading canonical utterances."""

    def test_load_canonical_utterances_array_format(
        self, service, sample_deployment, sample_utterances, temp_data_dir
    ):
        """Test loading utterances from array format JSON."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "canonical").mkdir(parents=True)

        # Write in array format
        transcript_data = [u.model_dump(mode="json") for u in sample_utterances]
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str)
        )

        utterances = service._load_canonical_utterances(
            sample_deployment, Path(temp_data_dir)
        )

        assert len(utterances) == 3
        assert utterances[0].text == "We need to check the battery levels before we start."

    def test_load_canonical_utterances_object_format(
        self, service, sample_deployment, sample_utterances, temp_data_dir
    ):
        """Test loading utterances from object format JSON."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "canonical").mkdir(parents=True)

        # Write in object format
        transcript_data = {
            "utterances": [u.model_dump(mode="json") for u in sample_utterances]
        }
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str)
        )

        utterances = service._load_canonical_utterances(
            sample_deployment, Path(temp_data_dir)
        )

        assert len(utterances) == 3

    def test_load_canonical_utterances_missing_file(
        self, service, sample_deployment, temp_data_dir
    ):
        """Test loading utterances when file doesn't exist."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "canonical").mkdir(parents=True)

        utterances = service._load_canonical_utterances(
            sample_deployment, Path(temp_data_dir)
        )

        assert utterances == []


class TestLoadPeopleNames:
    """Tests for loading people names."""

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

    def test_load_people_names_missing_file(self, service, temp_data_dir):
        """Test loading people names when file doesn't exist."""
        names = service._load_people_names(Path(temp_data_dir))
        assert names == {}
