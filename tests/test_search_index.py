"""Tests for the SearchIndexBuilder service."""

import shutil
import tempfile
from pathlib import Path

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.deployment import Deployment
from gram_deploy.models.canonical_utterance import CanonicalUtterance, UtteranceSource
from gram_deploy.models.event import DeploymentEvent, EventType
from gram_deploy.models.insight import DeploymentInsight, InsightType

# Import search_index directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "search_index",
    Path(__file__).parent.parent / "src" / "gram_deploy" / "services" / "search_index.py"
)
si_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si_module)
SearchIndexBuilder = si_module.SearchIndexBuilder
SearchResult = si_module.SearchResult
SearchResultType = si_module.SearchResultType


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for the search index."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def index_builder(temp_index_dir):
    """Create a SearchIndexBuilder with a temporary directory."""
    return SearchIndexBuilder(temp_index_dir)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
    )


@pytest.fixture
def sample_utterances():
    """Create sample utterances for testing."""
    deployment_id = "deploy:20250119_vinci_01"
    return [
        CanonicalUtterance(
            id=f"utterance:{deployment_id}/001",
            deployment_id=deployment_id,
            text="The Starlink battery is running low and needs to be replaced.",
            canonical_start_ms=0,
            canonical_end_ms=5000,
            speaker_id="person:damion",
            speaker_confidence=0.9,
            sources=[
                UtteranceSource(
                    source_id=f"source:{deployment_id}/gopro_01",
                    local_start_time=0.0,
                    local_end_time=5.0,
                )
            ],
        ),
        CanonicalUtterance(
            id=f"utterance:{deployment_id}/002",
            deployment_id=deployment_id,
            text="We need to check the network connectivity before proceeding.",
            canonical_start_ms=6000,
            canonical_end_ms=10000,
            speaker_id="person:chu",
            speaker_confidence=0.85,
            sources=[
                UtteranceSource(
                    source_id=f"source:{deployment_id}/gopro_01",
                    local_start_time=6.0,
                    local_end_time=10.0,
                )
            ],
        ),
        CanonicalUtterance(
            id=f"utterance:{deployment_id}/003",
            deployment_id=deployment_id,
            text="The solar panel output is looking good today.",
            canonical_start_ms=15000,
            canonical_end_ms=20000,
            speaker_id="person:damion",
            speaker_confidence=0.88,
            sources=[
                UtteranceSource(
                    source_id=f"source:{deployment_id}/phone_01",
                    local_start_time=15.0,
                    local_end_time=20.0,
                )
            ],
        ),
        CanonicalUtterance(
            id=f"utterance:{deployment_id}/004",
            deployment_id=deployment_id,
            text="Battery levels are critical, we need backup power.",
            canonical_start_ms=25000,
            canonical_end_ms=30000,
            speaker_id="person:chu",
            speaker_confidence=0.82,
            sources=[
                UtteranceSource(
                    source_id=f"source:{deployment_id}/gopro_01",
                    local_start_time=25.0,
                    local_end_time=30.0,
                )
            ],
        ),
    ]


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    deployment_id = "deploy:20250119_vinci_01"
    return [
        DeploymentEvent(
            id=f"event:{deployment_id}/001",
            deployment_id=deployment_id,
            event_type=EventType.ISSUE,
            canonical_time_ms=5000,
            title="Battery Low Warning",
            description="Starlink battery dropped below 20%",
        ),
        DeploymentEvent(
            id=f"event:{deployment_id}/002",
            deployment_id=deployment_id,
            event_type=EventType.MILESTONE,
            canonical_time_ms=20000,
            title="Network Established",
            description="Successfully connected to satellite network",
        ),
        DeploymentEvent(
            id=f"event:{deployment_id}/003",
            deployment_id=deployment_id,
            event_type=EventType.DECISION,
            canonical_time_ms=35000,
            title="Deploy Backup Power",
            description="Decision to activate backup battery system",
        ),
    ]


@pytest.fixture
def sample_insights():
    """Create sample insights for testing."""
    deployment_id = "deploy:20250119_vinci_01"
    return [
        DeploymentInsight(
            id=f"insight:{deployment_id}/001",
            deployment_id=deployment_id,
            insight_type=InsightType.RISK_IDENTIFIED,
            content="Battery management is a critical concern for extended deployments.",
            category="power_management",
        ),
        DeploymentInsight(
            id=f"insight:{deployment_id}/002",
            deployment_id=deployment_id,
            insight_type=InsightType.TECHNICAL_OBSERVATION,
            content="Solar panel efficiency varies significantly with weather conditions.",
            category="power_generation",
        ),
        DeploymentInsight(
            id=f"insight:{deployment_id}/003",
            deployment_id=deployment_id,
            insight_type=InsightType.LESSON_LEARNED,
            content="Network connectivity issues can be mitigated with proper antenna positioning.",
        ),
    ]


class TestIndexCreation:
    """Tests for index creation."""

    def test_init_creates_database(self, temp_index_dir):
        """Test that initializing the builder creates the database file."""
        builder = SearchIndexBuilder(temp_index_dir)

        db_path = Path(temp_index_dir) / "search.db"
        assert db_path.exists()

    def test_init_creates_fts_tables(self, index_builder):
        """Test that initialization creates the required FTS5 tables."""
        import sqlite3

        conn = sqlite3.connect(index_builder._db_path)
        cursor = conn.cursor()

        # Check utterances table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='utterances'")
        assert cursor.fetchone() is not None

        # Check events table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
        assert cursor.fetchone() is not None

        # Check insights table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='insights'")
        assert cursor.fetchone() is not None

        conn.close()

    def test_build_index_clears_existing_data(self, index_builder, sample_deployment, sample_utterances):
        """Test that build_index clears existing data for the deployment."""
        # Index some data
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["utterance_count"] == 4

        # Rebuild index - should clear data
        index_builder.build_index(sample_deployment)

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["utterance_count"] == 0

    def test_build_index_with_string_id(self, index_builder, sample_utterances):
        """Test that build_index works with a string deployment ID."""
        deployment_id = "deploy:20250119_vinci_01"

        index_builder.build_index(deployment_id)
        count = index_builder.index_utterances(deployment_id, sample_utterances)

        assert count == 4

    def test_index_utterances(self, index_builder, sample_deployment, sample_utterances):
        """Test indexing utterances."""
        index_builder.build_index(sample_deployment)
        count = index_builder.index_utterances(sample_deployment, sample_utterances)

        assert count == 4

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["utterance_count"] == 4
        assert stats["speaker_count"] == 2

    def test_index_events(self, index_builder, sample_deployment, sample_events):
        """Test indexing events."""
        index_builder.build_index(sample_deployment)
        count = index_builder.index_events(sample_deployment, sample_events)

        assert count == 3

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["event_count"] == 3

    def test_index_insights(self, index_builder, sample_deployment, sample_insights):
        """Test indexing insights."""
        index_builder.build_index(sample_deployment)
        count = index_builder.index_insights(sample_deployment, sample_insights)

        assert count == 3

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["insight_count"] == 3


class TestSearchQueries:
    """Tests for search functionality."""

    def test_search_utterances(self, index_builder, sample_deployment, sample_utterances):
        """Test searching utterances."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search("battery", sample_deployment.id)

        assert len(results) >= 2
        assert all(r.result_type == SearchResultType.UTTERANCE for r in results)
        assert all("battery" in r.text.lower() for r in results)

    def test_search_events(self, index_builder, sample_deployment, sample_events):
        """Test searching events."""
        index_builder.build_index(sample_deployment)
        index_builder.index_events(sample_deployment, sample_events)

        results = index_builder.search("network", sample_deployment.id)

        assert len(results) >= 1
        assert any(r.result_type == SearchResultType.EVENT for r in results)

    def test_search_insights(self, index_builder, sample_deployment, sample_insights):
        """Test searching insights."""
        index_builder.build_index(sample_deployment)
        index_builder.index_insights(sample_deployment, sample_insights)

        results = index_builder.search("battery", sample_deployment.id)

        assert len(results) >= 1
        assert any(r.result_type == SearchResultType.INSIGHT for r in results)

    def test_search_across_all_tables(
        self, index_builder, sample_deployment, sample_utterances, sample_events, sample_insights
    ):
        """Test searching across all indexed content types."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)
        index_builder.index_events(sample_deployment, sample_events)
        index_builder.index_insights(sample_deployment, sample_insights)

        results = index_builder.search("battery", sample_deployment.id)

        # Should find results from utterances, events, and insights
        result_types = {r.result_type for r in results}
        assert SearchResultType.UTTERANCE in result_types
        assert SearchResultType.EVENT in result_types
        assert SearchResultType.INSIGHT in result_types

    def test_search_no_results(self, index_builder, sample_deployment, sample_utterances):
        """Test searching with a query that has no matches."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search("xyznonexistent", sample_deployment.id)

        assert len(results) == 0

    def test_search_without_deployment_filter(
        self, index_builder, sample_utterances
    ):
        """Test searching without specifying a deployment ID."""
        # Index data for first deployment
        deployment_id_1 = "deploy:20250119_vinci_01"
        index_builder.build_index(deployment_id_1)
        index_builder.index_utterances(deployment_id_1, sample_utterances)

        # Create and index data for second deployment
        deployment_id_2 = "deploy:20250120_rome_01"
        index_builder.build_index(deployment_id_2)
        utterances_2 = [
            CanonicalUtterance(
                id=f"utterance:{deployment_id_2}/001",
                deployment_id=deployment_id_2,
                text="The battery system is working perfectly here.",
                canonical_start_ms=0,
                canonical_end_ms=5000,
                speaker_id="person:alice",
                speaker_confidence=0.9,
            )
        ]
        index_builder.index_utterances(deployment_id_2, utterances_2)

        # Search without deployment filter
        results = index_builder.search("battery")

        # Should find results from both deployments
        deployment_ids = {r.deployment_id for r in results}
        assert deployment_id_1 in deployment_ids
        assert deployment_id_2 in deployment_ids

    def test_search_results_have_snippets(self, index_builder, sample_deployment, sample_utterances):
        """Test that search results include highlighted snippets."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search("battery", sample_deployment.id)

        assert len(results) > 0
        # Snippets should contain the search term
        for result in results:
            assert result.snippet is not None

    def test_search_results_sorted_by_relevance(self, index_builder, sample_deployment, sample_utterances):
        """Test that search results are sorted by relevance score."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search("battery", sample_deployment.id)

        # Results should be sorted by relevance (descending)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score


class TestSearchFiltering:
    """Tests for filtered search functionality."""

    def test_search_by_speaker(self, index_builder, sample_deployment, sample_utterances):
        """Test filtering search results by speaker."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search_by_speaker("battery", "person:damion", sample_deployment.id)

        assert len(results) >= 1
        assert all(r.speaker_id == "person:damion" for r in results)

    def test_search_by_speaker_no_matches(self, index_builder, sample_deployment, sample_utterances):
        """Test search by speaker with no matching results."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        # Search for something Damion didn't say
        results = index_builder.search_by_speaker("network", "person:damion", sample_deployment.id)

        # Damion doesn't mention network, only Chu does
        assert len(results) == 0

    def test_search_by_speaker_across_deployments(self, index_builder, sample_utterances):
        """Test search by speaker without deployment filter."""
        deployment_id = "deploy:20250119_vinci_01"
        index_builder.build_index(deployment_id)
        index_builder.index_utterances(deployment_id, sample_utterances)

        results = index_builder.search_by_speaker("battery", "person:damion")

        assert len(results) >= 1
        assert all(r.speaker_id == "person:damion" for r in results)

    def test_search_by_timerange(self, index_builder, sample_deployment, sample_utterances):
        """Test filtering search results by time range."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        # Search for "battery" in the first 10 seconds
        results = index_builder.search_by_timerange("battery", 0, 10000, sample_deployment.id)

        assert len(results) >= 1
        # All utterance results should be within the time range
        for result in results:
            if result.result_type == SearchResultType.UTTERANCE:
                assert result.canonical_time_ms is not None
                assert result.canonical_time_ms >= 0
                assert result.canonical_time_ms <= 10000

    def test_search_by_timerange_no_matches(self, index_builder, sample_deployment, sample_utterances):
        """Test search by time range with no results."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        # Search for "battery" in a time range where it wasn't mentioned
        results = index_builder.search_by_timerange("battery", 11000, 14000, sample_deployment.id)

        # No utterances mentioning battery in this range
        utterance_results = [r for r in results if r.result_type == SearchResultType.UTTERANCE]
        assert len(utterance_results) == 0

    def test_search_by_timerange_includes_events(
        self, index_builder, sample_deployment, sample_utterances, sample_events
    ):
        """Test that time range search includes events."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)
        index_builder.index_events(sample_deployment, sample_events)

        # Search for "battery" in a range that includes the Battery Low Warning event
        results = index_builder.search_by_timerange("battery", 0, 10000, sample_deployment.id)

        event_results = [r for r in results if r.result_type == SearchResultType.EVENT]
        assert len(event_results) >= 1


class TestDeploymentManagement:
    """Tests for deployment management functionality."""

    def test_get_deployment_stats(self, index_builder, sample_deployment, sample_utterances, sample_events):
        """Test getting statistics for a deployment."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)
        index_builder.index_events(sample_deployment, sample_events)

        stats = index_builder.get_deployment_stats(sample_deployment.id)

        assert stats["utterance_count"] == 4
        assert stats["event_count"] == 3
        assert stats["speaker_count"] == 2

    def test_get_deployment_stats_nonexistent(self, index_builder):
        """Test getting stats for a non-existent deployment."""
        stats = index_builder.get_deployment_stats("deploy:nonexistent")

        assert stats["utterance_count"] == 0
        assert stats["event_count"] == 0
        assert stats["insight_count"] == 0
        assert stats["speaker_count"] == 0

    def test_delete_deployment_index(
        self, index_builder, sample_deployment, sample_utterances, sample_events, sample_insights
    ):
        """Test deleting a deployment's index."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)
        index_builder.index_events(sample_deployment, sample_events)
        index_builder.index_insights(sample_deployment, sample_insights)

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["utterance_count"] == 4

        index_builder.delete_deployment_index(sample_deployment.id)

        stats = index_builder.get_deployment_stats(sample_deployment.id)
        assert stats["utterance_count"] == 0
        assert stats["event_count"] == 0
        assert stats["insight_count"] == 0

    def test_list_indexed_deployments(self, index_builder, sample_utterances):
        """Test listing all indexed deployments."""
        # Index two deployments
        deployment_id_1 = "deploy:20250119_vinci_01"
        deployment_id_2 = "deploy:20250120_rome_01"

        index_builder.build_index(deployment_id_1)
        index_builder.index_utterances(deployment_id_1, sample_utterances)

        index_builder.build_index(deployment_id_2)
        index_builder.index_utterances(deployment_id_2, sample_utterances[:2])

        deployments = index_builder.list_indexed_deployments()

        assert len(deployments) == 2
        deployment_ids = {d["deployment_id"] for d in deployments}
        assert deployment_id_1 in deployment_ids
        assert deployment_id_2 in deployment_ids


class TestSearchResultProperties:
    """Tests for SearchResult properties and types."""

    def test_utterance_result_has_speaker_id(self, index_builder, sample_deployment, sample_utterances):
        """Test that utterance results include speaker_id."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search("battery", sample_deployment.id)
        utterance_results = [r for r in results if r.result_type == SearchResultType.UTTERANCE]

        assert len(utterance_results) > 0
        for result in utterance_results:
            assert result.speaker_id is not None

    def test_utterance_result_has_time_range(self, index_builder, sample_deployment, sample_utterances):
        """Test that utterance results include canonical start and end times."""
        index_builder.build_index(sample_deployment)
        index_builder.index_utterances(sample_deployment, sample_utterances)

        results = index_builder.search("Starlink", sample_deployment.id)
        utterance_results = [r for r in results if r.result_type == SearchResultType.UTTERANCE]

        assert len(utterance_results) > 0
        for result in utterance_results:
            assert result.canonical_time_ms is not None
            assert result.canonical_end_ms is not None
            assert result.canonical_time_ms <= result.canonical_end_ms

    def test_event_result_has_event_type(self, index_builder, sample_deployment, sample_events):
        """Test that event results include event_type."""
        index_builder.build_index(sample_deployment)
        index_builder.index_events(sample_deployment, sample_events)

        results = index_builder.search("battery", sample_deployment.id)
        event_results = [r for r in results if r.result_type == SearchResultType.EVENT]

        assert len(event_results) > 0
        for result in event_results:
            assert result.event_type is not None

    def test_insight_result_has_category(self, index_builder, sample_deployment, sample_insights):
        """Test that insight results include category."""
        index_builder.build_index(sample_deployment)
        index_builder.index_insights(sample_deployment, sample_insights)

        results = index_builder.search("battery", sample_deployment.id)
        insight_results = [r for r in results if r.result_type == SearchResultType.INSIGHT]

        assert len(insight_results) > 0
        for result in insight_results:
            assert result.category is not None
