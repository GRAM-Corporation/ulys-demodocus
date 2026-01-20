"""Tests for the DeploymentManager service."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
from pathlib import Path

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly from modules to avoid loading all services
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.source import Source, DeviceType

# Import deployment_manager directly to avoid services/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "deployment_manager",
    Path(__file__).parent.parent / "src" / "gram_deploy" / "services" / "deployment_manager.py"
)
dm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm_module)
DeploymentManager = dm_module.DeploymentManager


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def manager(temp_data_dir):
    """Create a DeploymentManager with a temp data directory."""
    return DeploymentManager(temp_data_dir)


class TestCreateDeployment:
    """Tests for create_deployment method."""

    def test_create_deployment_basic(self, manager, temp_data_dir):
        """Test creating a deployment with basic parameters."""
        deployment = manager.create_deployment(
            location="vinci",
            date="2025-01-19",
        )

        assert deployment.id == "deploy:20250119_vinci_01"
        assert deployment.location == "vinci"
        assert deployment.date == "2025-01-19"
        assert deployment.status == DeploymentStatus.INGESTING
        assert deployment.team_members == []
        assert deployment.notes is None

    def test_create_deployment_with_all_parameters(self, manager):
        """Test creating a deployment with all parameters."""
        deployment = manager.create_deployment(
            location="test site",
            date="2025-01-20",
            team_members=["person:alice", "person:bob"],
            notes="Test deployment for unit tests",
        )

        assert deployment.id == "deploy:20250120_test_site_01"
        assert deployment.location == "test site"
        assert deployment.team_members == ["person:alice", "person:bob"]
        assert deployment.notes == "Test deployment for unit tests"

    def test_create_deployment_sequential_numbering(self, manager):
        """Test that deployments get sequential numbers."""
        d1 = manager.create_deployment(location="vinci", date="2025-01-19")
        d2 = manager.create_deployment(location="vinci", date="2025-01-19")
        d3 = manager.create_deployment(location="vinci", date="2025-01-19")

        assert d1.id == "deploy:20250119_vinci_01"
        assert d2.id == "deploy:20250119_vinci_02"
        assert d3.id == "deploy:20250119_vinci_03"

    def test_create_deployment_different_locations(self, manager):
        """Test that different locations get their own sequences."""
        d1 = manager.create_deployment(location="vinci", date="2025-01-19")
        d2 = manager.create_deployment(location="rome", date="2025-01-19")
        d3 = manager.create_deployment(location="vinci", date="2025-01-19")

        assert d1.id == "deploy:20250119_vinci_01"
        assert d2.id == "deploy:20250119_rome_01"
        assert d3.id == "deploy:20250119_vinci_02"

    def test_create_deployment_creates_directory_structure(self, manager, temp_data_dir):
        """Test that create_deployment creates the expected directories."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        assert deploy_dir.exists()
        assert (deploy_dir / "sources").exists()
        assert (deploy_dir / "outputs").exists()
        assert (deploy_dir / "canonical").exists()
        assert (deploy_dir / "analysis").exists()
        assert (deploy_dir / "cache").exists()

    def test_create_deployment_writes_json(self, manager, temp_data_dir):
        """Test that create_deployment writes deployment.json."""
        deployment = manager.create_deployment(
            location="vinci",
            date="2025-01-19",
            notes="Test notes",
        )

        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deployment_json = deploy_dir / "deployment.json"
        assert deployment_json.exists()

        data = json.loads(deployment_json.read_text())
        assert data["id"] == "deploy:20250119_vinci_01"
        assert data["location"] == "vinci"
        assert data["notes"] == "Test notes"

    def test_create_deployment_updates_index(self, manager, temp_data_dir):
        """Test that create_deployment updates the index file."""
        manager.create_deployment(location="vinci", date="2025-01-19")
        manager.create_deployment(location="rome", date="2025-01-20")

        index_path = Path(temp_data_dir) / "index.json"
        index = json.loads(index_path.read_text())

        assert "deploy:20250119_vinci_01" in index["deployments"]
        assert "deploy:20250120_rome_01" in index["deployments"]


class TestLoadDeployment:
    """Tests for load_deployment method."""

    def test_load_deployment_success(self, manager):
        """Test loading an existing deployment."""
        created = manager.create_deployment(location="vinci", date="2025-01-19")
        loaded = manager.load_deployment(created.id)

        assert loaded.id == created.id
        assert loaded.location == created.location
        assert loaded.date == created.date

    def test_load_deployment_not_found(self, manager):
        """Test that load_deployment raises for non-existent deployment."""
        with pytest.raises(ValueError, match="Deployment not found"):
            manager.load_deployment("deploy:20250119_nonexistent_01")

    def test_get_deployment_returns_none(self, manager):
        """Test that get_deployment returns None for non-existent deployment."""
        result = manager.get_deployment("deploy:20250119_nonexistent_01")
        assert result is None


class TestSaveDeployment:
    """Tests for save_deployment method."""

    def test_save_deployment_updates_fields(self, manager):
        """Test that save_deployment persists changes."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")
        deployment.notes = "Updated notes"
        deployment.team_members = ["person:alice"]
        manager.save_deployment(deployment)

        loaded = manager.load_deployment(deployment.id)
        assert loaded.notes == "Updated notes"
        assert loaded.team_members == ["person:alice"]

    def test_save_deployment_updates_timestamp(self, manager):
        """Test that save_deployment updates updated_at timestamp."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")
        original_updated = deployment.updated_at

        deployment.notes = "New notes"
        manager.save_deployment(deployment)

        loaded = manager.load_deployment(deployment.id)
        assert loaded.updated_at >= original_updated


class TestAddSource:
    """Tests for add_source method."""

    def test_add_source_basic(self, manager, temp_data_dir):
        """Test adding a source without actual files (mocked ffprobe)."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        # Create a dummy video file
        dummy_video = Path(temp_data_dir) / "test_video.mp4"
        dummy_video.write_bytes(b"fake video data")

        with patch.object(manager, "_extract_video_metadata") as mock_extract:
            mock_extract.return_value = {
                "duration": 120.5,
                "video_codec": "h264",
                "audio_codec": "aac",
                "resolution": "1920x1080",
                "fps": 30.0,
            }

            source = manager.add_source(
                deployment_id=deployment.id,
                device_type="gopro",
                device_number=1,
                file_paths=[str(dummy_video)],
            )

        assert source.id == "source:deploy:20250119_vinci_01/gopro_01"
        assert source.device_type == DeviceType.GOPRO
        assert source.device_number == 1
        assert len(source.files) == 1
        assert source.files[0].duration_seconds == 120.5

    def test_add_source_updates_deployment(self, manager, temp_data_dir):
        """Test that add_source updates the deployment's sources list."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        dummy_video = Path(temp_data_dir) / "test_video.mp4"
        dummy_video.write_bytes(b"fake video data")

        with patch.object(manager, "_extract_video_metadata") as mock_extract:
            mock_extract.return_value = {"duration": 60.0}
            manager.add_source(
                deployment_id=deployment.id,
                device_type="phone",
                device_number=1,
                file_paths=[str(dummy_video)],
            )

        loaded = manager.load_deployment(deployment.id)
        assert "source:deploy:20250119_vinci_01/phone_01" in loaded.sources

    def test_add_source_creates_directory(self, manager, temp_data_dir):
        """Test that add_source creates source directory."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        dummy_video = Path(temp_data_dir) / "test_video.mp4"
        dummy_video.write_bytes(b"fake video data")

        with patch.object(manager, "_extract_video_metadata") as mock_extract:
            mock_extract.return_value = {"duration": 60.0}
            manager.add_source(
                deployment_id=deployment.id,
                device_type="gopro",
                device_number=2,
                file_paths=[str(dummy_video)],
            )

        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        source_dir = deploy_dir / "sources" / "gopro_02"
        assert source_dir.exists()
        assert (source_dir / "source.json").exists()

    def test_add_source_invalid_deployment(self, manager, temp_data_dir):
        """Test that add_source raises for non-existent deployment."""
        with pytest.raises(ValueError, match="Deployment not found"):
            manager.add_source(
                deployment_id="deploy:20250119_nonexistent_01",
                device_type="gopro",
                device_number=1,
                file_paths=[],
            )

    def test_add_source_multiple_files(self, manager, temp_data_dir):
        """Test adding a source with multiple video files."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        # Create multiple dummy video files
        videos = []
        for i in range(3):
            video = Path(temp_data_dir) / f"GX01000{i}.MP4"
            video.write_bytes(b"fake video data")
            videos.append(str(video))

        with patch.object(manager, "_extract_video_metadata") as mock_extract:
            mock_extract.return_value = {"duration": 300.0}  # 5 minutes each
            source = manager.add_source(
                deployment_id=deployment.id,
                device_type="gopro",
                device_number=1,
                file_paths=videos,
            )

        assert len(source.files) == 3
        assert source.total_duration_seconds == 900.0  # 15 minutes total
        # Check sequential offsets
        assert source.files[0].start_offset_ms == 0
        assert source.files[1].start_offset_ms == 300000
        assert source.files[2].start_offset_ms == 600000


class TestListDeployments:
    """Tests for list_deployments method."""

    def test_list_deployments_empty(self, manager):
        """Test list_deployments with no deployments."""
        result = manager.list_deployments()
        assert result == []

    def test_list_deployments_returns_ids(self, manager):
        """Test that list_deployments returns ID strings."""
        manager.create_deployment(location="vinci", date="2025-01-19")
        manager.create_deployment(location="rome", date="2025-01-20")

        result = manager.list_deployments()

        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)
        assert "deploy:20250119_vinci_01" in result
        assert "deploy:20250120_rome_01" in result

    def test_list_deployments_sorted_reverse_chronologically(self, manager):
        """Test that list_deployments returns IDs in reverse chronological order."""
        manager.create_deployment(location="site1", date="2025-01-15")
        manager.create_deployment(location="site2", date="2025-01-20")
        manager.create_deployment(location="site3", date="2025-01-18")

        result = manager.list_deployments()

        # Should be newest first
        assert result[0] == "deploy:20250120_site2_01"
        assert result[1] == "deploy:20250118_site3_01"
        assert result[2] == "deploy:20250115_site1_01"

    def test_get_deployments_returns_deployment_objects(self, manager):
        """Test that get_deployments returns Deployment objects."""
        manager.create_deployment(location="vinci", date="2025-01-19")

        result = manager.get_deployments()

        assert isinstance(result, list)
        assert all(isinstance(d, Deployment) for d in result)
        assert result[0].id == "deploy:20250119_vinci_01"


class TestUpdateStatus:
    """Tests for update_status method."""

    def test_update_status_basic(self, manager):
        """Test updating deployment status."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")
        assert deployment.status == DeploymentStatus.INGESTING

        manager.update_status(deployment.id, "transcribing")

        loaded = manager.load_deployment(deployment.id)
        assert loaded.status == DeploymentStatus.TRANSCRIBING

    def test_update_status_with_checkpoint(self, manager):
        """Test updating status with checkpoint."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        manager.update_status(
            deployment.id,
            "aligning",
            checkpoint="transcription_complete",
        )

        loaded = manager.load_deployment(deployment.id)
        assert loaded.status == DeploymentStatus.ALIGNING
        assert loaded.checkpoint == "transcription_complete"

    def test_update_status_with_error(self, manager):
        """Test updating status with error message."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        manager.update_status(
            deployment.id,
            "failed",
            error="Transcription API timeout",
        )

        loaded = manager.load_deployment(deployment.id)
        assert loaded.status == DeploymentStatus.FAILED
        assert loaded.error_message == "Transcription API timeout"

    def test_update_status_invalid_deployment(self, manager):
        """Test that update_status raises for non-existent deployment."""
        with pytest.raises(ValueError, match="Deployment not found"):
            manager.update_status("deploy:20250119_nonexistent_01", "complete")


class TestGetSource:
    """Tests for get_source and get_sources methods."""

    def test_get_source_success(self, manager, temp_data_dir):
        """Test retrieving a source by ID."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        dummy_video = Path(temp_data_dir) / "test.mp4"
        dummy_video.write_bytes(b"fake")

        with patch.object(manager, "_extract_video_metadata") as mock:
            mock.return_value = {"duration": 60.0}
            created = manager.add_source(
                deployment.id, "gopro", 1, [str(dummy_video)]
            )

        retrieved = manager.get_source(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_source_not_found(self, manager):
        """Test that get_source returns None for non-existent source."""
        result = manager.get_source("source:deploy:20250119_vinci_01/gopro_99")
        assert result is None

    def test_get_sources_returns_all(self, manager, temp_data_dir):
        """Test retrieving all sources for a deployment."""
        deployment = manager.create_deployment(location="vinci", date="2025-01-19")

        dummy_video = Path(temp_data_dir) / "test.mp4"
        dummy_video.write_bytes(b"fake")

        with patch.object(manager, "_extract_video_metadata") as mock:
            mock.return_value = {"duration": 60.0}
            manager.add_source(deployment.id, "gopro", 1, [str(dummy_video)])
            manager.add_source(deployment.id, "phone", 1, [str(dummy_video)])

        sources = manager.get_sources(deployment.id)
        assert len(sources) == 2


class TestExtractVideoMetadata:
    """Tests for _extract_video_metadata method."""

    def test_extract_metadata_ffprobe_success(self, manager):
        """Test metadata extraction with successful ffprobe call."""
        mock_ffprobe_output = {
            "format": {"duration": "300.5"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30000/1001",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                },
            ],
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_ffprobe_output),
            )

            metadata = manager._extract_video_metadata("/path/to/video.mp4")

        assert metadata["duration"] == 300.5
        assert metadata["video_codec"] == "h264"
        assert metadata["audio_codec"] == "aac"
        assert metadata["resolution"] == "1920x1080"
        assert metadata["fps"] == 29.97

    def test_extract_metadata_ffprobe_failure(self, manager):
        """Test metadata extraction when ffprobe fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            metadata = manager._extract_video_metadata("/path/to/video.mp4")

        assert metadata == {"duration": 0.0}
