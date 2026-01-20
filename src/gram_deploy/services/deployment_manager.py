"""Deployment Manager - handles creation, configuration, and lifecycle of deployments.

Responsible for:
- Creating deployments with correct directory structure
- Adding video sources with metadata extraction
- Managing deployment status and checkpoints
- Persisting deployment state to JSON files
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from gram_deploy.models import (
    Deployment,
    DeploymentStatus,
    DeviceType,
    Source,
    SourceFile,
    TranscriptStatus,
)


class DeploymentManager:
    """Manages deployment lifecycle and storage."""

    def __init__(self, data_dir: str):
        """Initialize the manager with the root data directory.

        Args:
            data_dir: Root directory for deployment data (e.g., ./deployments)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Ensure the deployment index file exists."""
        index_path = self.data_dir / "index.json"
        if not index_path.exists():
            index_path.write_text(json.dumps({"deployments": []}, indent=2))

    def _get_deployment_dir(self, deployment_id: str) -> Path:
        """Get the directory path for a deployment."""
        # Convert deploy:20250119_vinci_01 to deploy_20250119_vinci_01
        dir_name = deployment_id.replace(":", "_")
        return self.data_dir / dir_name

    def _get_next_sequence(self, location: str, date: str) -> int:
        """Get the next sequence number for a location/date combination."""
        date_compact = date.replace("-", "")
        location_slug = location.lower().replace(" ", "_")
        prefix = f"deploy:{date_compact}_{location_slug}_"

        existing = self.list_deployments()
        sequences = [
            int(d.split("_")[-1])
            for d in existing
            if d.startswith(prefix)
        ]
        return max(sequences, default=0) + 1

    def create_deployment(
        self,
        location: str,
        date: str,
        team_members: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> Deployment:
        """Create a new deployment.

        Args:
            location: Human-readable location name
            date: ISO 8601 date (YYYY-MM-DD)
            team_members: Optional list of person IDs for team members present
            notes: Optional free-form notes

        Returns:
            The created Deployment entity
        """
        sequence = self._get_next_sequence(location, date)
        deployment_id = Deployment.generate_id(location, date, sequence)

        deployment = Deployment(
            id=deployment_id,
            location=location,
            date=date,
            team_members=team_members or [],
            notes=notes,
            status=DeploymentStatus.INGESTING,
        )

        # Create directory structure
        deploy_dir = self._get_deployment_dir(deployment_id)
        (deploy_dir / "sources").mkdir(parents=True, exist_ok=True)
        (deploy_dir / "canonical").mkdir(exist_ok=True)
        (deploy_dir / "analysis").mkdir(exist_ok=True)
        (deploy_dir / "analysis" / "search_index").mkdir(exist_ok=True)
        (deploy_dir / "outputs").mkdir(exist_ok=True)
        (deploy_dir / "cache").mkdir(exist_ok=True)
        (deploy_dir / "cache" / "audio_extracts").mkdir(exist_ok=True)
        (deploy_dir / "cache" / "alignment").mkdir(exist_ok=True)
        (deploy_dir / "cache" / "llm_responses").mkdir(exist_ok=True)

        # Save deployment
        self._save_deployment(deployment)
        self._update_index(deployment)

        return deployment

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Retrieve a deployment by ID.

        Args:
            deployment_id: The deployment ID

        Returns:
            The Deployment entity, or None if not found
        """
        deploy_dir = self._get_deployment_dir(deployment_id)
        deployment_path = deploy_dir / "deployment.json"

        if not deployment_path.exists():
            return None

        data = json.loads(deployment_path.read_text())
        return Deployment.model_validate(data)

    def load_deployment(self, deployment_id: str) -> Deployment:
        """Load a deployment by ID.

        This method raises an exception if the deployment is not found,
        unlike get_deployment which returns None.

        Args:
            deployment_id: The deployment ID

        Returns:
            The Deployment entity

        Raises:
            ValueError: If the deployment is not found
        """
        deployment = self.get_deployment(deployment_id)
        if deployment is None:
            raise ValueError(f"Deployment not found: {deployment_id}")
        return deployment

    def save_deployment(self, deployment: Deployment) -> None:
        """Save a deployment to disk.

        Updates the updated_at timestamp and persists the deployment.

        Args:
            deployment: The Deployment entity to save
        """
        deployment.updated_at = datetime.utcnow()
        self._save_deployment(deployment)

    def list_deployments(self) -> list[str]:
        """List all deployment IDs.

        Reads the deployment index and returns all deployment IDs
        in reverse chronological order.

        Returns:
            List of deployment ID strings
        """
        index_path = self.data_dir / "index.json"
        index = json.loads(index_path.read_text())

        deployment_ids = index.get("deployments", [])
        # Sort by date descending (ID contains date)
        deployment_ids.sort(reverse=True)

        return deployment_ids

    def get_deployments(self, limit: int = 50, offset: int = 0) -> list[Deployment]:
        """Get deployments in reverse chronological order.

        Args:
            limit: Maximum number of deployments to return
            offset: Number of deployments to skip

        Returns:
            List of Deployment entities
        """
        deployment_ids = self.list_deployments()

        deployments = []
        for did in deployment_ids[offset:offset + limit]:
            deployment = self.get_deployment(did)
            if deployment:
                deployments.append(deployment)

        return deployments

    def update_deployment(self, deployment: Deployment) -> None:
        """Save changes to a deployment.

        Args:
            deployment: The Deployment entity to save
        """
        deployment.updated_at = datetime.utcnow()
        self._save_deployment(deployment)

    def add_source(
        self,
        deployment_id: str,
        device_type: str,
        device_number: int,
        file_paths: list[str],
        device_model: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> Source:
        """Add a video source to a deployment.

        Args:
            deployment_id: The deployment ID
            device_type: Type of device (gopro, phone, fixed, drone, other)
            device_number: Device number (1, 2, etc.)
            file_paths: Paths to video files
            device_model: Optional specific model name
            operator: Optional person ID of camera operator

        Returns:
            The created Source entity
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")

        device_type_enum = DeviceType(device_type)
        source_id = Source.generate_id(deployment_id, device_type_enum, device_number)

        # Create source directory
        deploy_dir = self._get_deployment_dir(deployment_id)
        source_dir = deploy_dir / "sources" / f"{device_type}_{device_number:02d}"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "files").mkdir(exist_ok=True)

        # Process files and extract metadata
        source_files = []
        total_duration = 0.0
        current_offset = 0

        # Sort files by name (handles GoPro naming: GX010001.MP4, GX010002.MP4)
        sorted_paths = sorted(file_paths)

        for file_path in sorted_paths:
            path = Path(file_path)
            metadata = self._extract_video_metadata(str(path))

            duration = metadata.get("duration", 0.0)
            source_file = SourceFile(
                filename=path.name,
                file_path=str(path.absolute()),
                file_size_bytes=path.stat().st_size if path.exists() else None,
                duration_seconds=duration,
                start_offset_ms=current_offset,
                video_codec=metadata.get("video_codec"),
                audio_codec=metadata.get("audio_codec"),
                resolution=metadata.get("resolution"),
                fps=metadata.get("fps"),
            )
            source_files.append(source_file)

            current_offset += int(duration * 1000)
            total_duration += duration

        source = Source(
            id=source_id,
            deployment_id=deployment_id,
            device_type=device_type_enum,
            device_number=device_number,
            device_model=device_model,
            operator=operator,
            files=source_files,
            total_duration_seconds=total_duration,
            transcript_status=TranscriptStatus.PENDING,
        )

        # Save source
        source_path = source_dir / "source.json"
        source_path.write_text(source.model_dump_json(indent=2))

        # Update deployment
        if source_id not in deployment.sources:
            deployment.sources.append(source_id)
            self.update_deployment(deployment)

        return source

    def get_source(self, source_id: str) -> Optional[Source]:
        """Retrieve a source by ID.

        Args:
            source_id: The source ID

        Returns:
            The Source entity, or None if not found
        """
        # Parse source ID: source:deploy:20250119_vinci_01/gopro_01
        parts = source_id.replace("source:", "").split("/")
        if len(parts) != 2:
            return None

        deployment_id = parts[0]  # Already includes "deploy:" prefix
        device_part = parts[1]  # gopro_01

        deploy_dir = self._get_deployment_dir(deployment_id)
        source_path = deploy_dir / "sources" / device_part / "source.json"

        if not source_path.exists():
            return None

        data = json.loads(source_path.read_text())
        return Source.model_validate(data)

    def get_sources(self, deployment_id: str) -> list[Source]:
        """Retrieve all sources for a deployment.

        Args:
            deployment_id: The deployment ID

        Returns:
            List of Source entities
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return []

        sources = []
        for source_id in deployment.sources:
            source = self.get_source(source_id)
            if source:
                sources.append(source)

        return sources

    def set_deployment_status(
        self,
        deployment_id: str,
        status: str,
        checkpoint: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the processing status of a deployment.

        Args:
            deployment_id: The deployment ID
            status: New status value
            checkpoint: Optional checkpoint name
            error_message: Optional error message if status is FAILED
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")

        deployment.status = DeploymentStatus(status)
        if checkpoint:
            deployment.checkpoint = checkpoint
        if error_message:
            deployment.error_message = error_message

        self.update_deployment(deployment)

    def update_status(
        self,
        deployment_id: str,
        status: str,
        checkpoint: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the processing status of a deployment.

        This is an alias for set_deployment_status with a slightly different
        parameter name (error instead of error_message) to match the spec.

        Args:
            deployment_id: The deployment ID
            status: New status value
            checkpoint: Optional checkpoint name
            error: Optional error message if status is FAILED
        """
        self.set_deployment_status(
            deployment_id=deployment_id,
            status=status,
            checkpoint=checkpoint,
            error_message=error,
        )

    def _save_deployment(self, deployment: Deployment) -> None:
        """Save deployment to disk."""
        deploy_dir = self._get_deployment_dir(deployment.id)
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deployment_path = deploy_dir / "deployment.json"
        deployment_path.write_text(deployment.model_dump_json(indent=2))

    def _update_index(self, deployment: Deployment) -> None:
        """Update the deployment index."""
        index_path = self.data_dir / "index.json"
        index = json.loads(index_path.read_text())

        if deployment.id not in index["deployments"]:
            index["deployments"].append(deployment.id)
            index_path.write_text(json.dumps(index, indent=2))

    def _extract_video_metadata(self, video_path: str) -> dict:
        """Extract metadata from a video file using ffprobe.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with duration, codecs, resolution, fps
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return {"duration": 0.0}

            data = json.loads(result.stdout)
            metadata = {}

            # Extract duration
            if "format" in data:
                metadata["duration"] = float(data["format"].get("duration", 0))

            # Extract stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    metadata["video_codec"] = stream.get("codec_name")
                    width = stream.get("width")
                    height = stream.get("height")
                    if width and height:
                        metadata["resolution"] = f"{width}x{height}"
                    # Parse fps from r_frame_rate (e.g., "30000/1001")
                    fps_str = stream.get("r_frame_rate", "")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        if int(den) > 0:
                            metadata["fps"] = round(int(num) / int(den), 2)
                elif stream.get("codec_type") == "audio":
                    metadata["audio_codec"] = stream.get("codec_name")

            return metadata

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            return {"duration": 0.0}
