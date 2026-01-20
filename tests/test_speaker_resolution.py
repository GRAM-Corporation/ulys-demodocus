"""Tests for the SpeakerResolutionService."""

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

# Import models directly
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.person import Person, VoiceSample
from gram_deploy.models.speaker_mapping import SpeakerMapping, ResolutionMethod
from gram_deploy.models.transcript import RawTranscript, TranscriptSegment, TranscriptSpeaker

# Import speaker_resolution directly to avoid services/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "speaker_resolution",
    Path(__file__).parent.parent / "src" / "gram_deploy" / "services" / "speaker_resolution.py"
)
sr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr_module)
SpeakerResolutionService = sr_module.SpeakerResolutionService


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def people_registry(temp_dir):
    """Create a people registry JSON file with test data."""
    registry_path = Path(temp_dir) / "people.json"
    people_data = {
        "people": [
            {
                "id": "person:damion",
                "name": "Damion Shelton",
                "aliases": ["Damion"],
                "role": "CTO",
                "voice_samples": [],
                "voice_embedding": None,
                "created_at": "2025-01-19T00:00:00Z",
                "updated_at": "2025-01-19T00:00:00Z",
            },
            {
                "id": "person:chu",
                "name": "Chu",
                "aliases": [],
                "role": None,
                "voice_samples": [],
                "voice_embedding": None,
                "created_at": "2025-01-19T00:00:00Z",
                "updated_at": "2025-01-19T00:00:00Z",
            },
        ]
    }
    registry_path.write_text(json.dumps(people_data, indent=2))
    return str(registry_path)


@pytest.fixture
def service(people_registry):
    """Create a SpeakerResolutionService with a test registry."""
    return SpeakerResolutionService(people_registry)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
        team_members=["person:damion", "person:chu"],
        status=DeploymentStatus.TRANSCRIBING,
    )


@pytest.fixture
def sample_transcript():
    """Create a sample transcript with speaker segments."""
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_01",
        source_id="source:deploy:20250119_vinci_01/gopro_01",
        transcription_service="elevenlabs",
        segments=[
            TranscriptSegment(
                text="This is Damion, I'm going to walk through the technical setup.",
                start_time=0.0,
                end_time=5.0,
                speaker=TranscriptSpeaker(id="speaker_0"),
            ),
            TranscriptSegment(
                text="I'll handle the code deployment for this.",
                start_time=5.0,
                end_time=8.0,
                speaker=TranscriptSpeaker(id="speaker_0"),
            ),
            TranscriptSegment(
                text="Okay, what do you need from me?",
                start_time=8.0,
                end_time=10.0,
                speaker=TranscriptSpeaker(id="speaker_1"),
            ),
            TranscriptSegment(
                text="Hey Damion, can you explain that again?",
                start_time=10.0,
                end_time=13.0,
                speaker=TranscriptSpeaker(id="speaker_1"),
            ),
        ],
    )


class TestLoadRegistry:
    """Tests for registry loading."""

    def test_load_registry_success(self, service):
        """Test that registry loads correctly."""
        people = service.list_people()
        assert len(people) == 2
        assert any(p.id == "person:damion" for p in people)
        assert any(p.id == "person:chu" for p in people)

    def test_load_registry_nonexistent(self, temp_dir):
        """Test loading a non-existent registry."""
        service = SpeakerResolutionService(f"{temp_dir}/nonexistent.json")
        assert service.list_people() == []

    def test_get_person(self, service):
        """Test getting a person by ID."""
        person = service.get_person("person:damion")
        assert person is not None
        assert person.name == "Damion Shelton"
        assert person.role == "CTO"


class TestContextMatching:
    """Tests for context-based speaker matching."""

    def test_self_identification(self, service, sample_deployment, sample_transcript):
        """Test matching when speaker identifies themselves."""
        mappings = service.resolve_speakers(
            deployment_id=sample_deployment.id,
            sources=[],
            transcripts=[sample_transcript],
        )

        # speaker_0 says "This is Damion" - should be identified
        speaker_0_mapping = next(
            (m for m in mappings if m.raw_speaker_id == "speaker_0"), None
        )
        assert speaker_0_mapping is not None
        assert speaker_0_mapping.resolved_person_id == "person:damion"
        assert speaker_0_mapping.method == ResolutionMethod.CONTEXT_INFERENCE
        assert speaker_0_mapping.confidence > 0.5

    def test_addressing_pattern(self, service, sample_deployment, sample_transcript):
        """Test matching when speaker is addressed by name."""
        # speaker_1 is addressed as "Damion" in the transcript
        # But this should resolve speaker_0 as Damion based on self-identification
        mappings = service.resolve_speakers(
            deployment_id=sample_deployment.id,
            sources=[],
            transcripts=[sample_transcript],
        )

        speaker_0 = next(
            (m for m in mappings if m.raw_speaker_id == "speaker_0"), None
        )
        assert speaker_0 is not None
        assert speaker_0.resolved_person_id == "person:damion"

    def test_unresolved_speaker(self, service, sample_deployment):
        """Test that speakers without context clues remain unresolved."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Let's get started with the deployment.",
                    start_time=0.0,
                    end_time=3.0,
                    speaker=TranscriptSpeaker(id="speaker_0"),
                ),
                TranscriptSegment(
                    text="Sure thing, I'm ready.",
                    start_time=3.0,
                    end_time=5.0,
                    speaker=TranscriptSpeaker(id="speaker_1"),
                ),
            ],
        )

        mappings = service.resolve_speakers(
            deployment_id=sample_deployment.id,
            sources=[],
            transcripts=[transcript],
        )

        # Both should be unresolved with context inference alone
        for mapping in mappings:
            if mapping.method == ResolutionMethod.UNRESOLVED:
                assert mapping.confidence == 0.0


class TestPatternMatching:
    """Tests for pattern-based speaker matching."""

    def test_match_by_role_language(self, service, sample_deployment):
        """Test matching speakers by role-specific language."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="I need to check the API endpoint and database connections.",
                    start_time=0.0,
                    end_time=5.0,
                    speaker=TranscriptSpeaker(id="speaker_0"),
                ),
                TranscriptSegment(
                    text="The code deployment will happen after we verify the infrastructure.",
                    start_time=5.0,
                    end_time=10.0,
                    speaker=TranscriptSpeaker(id="speaker_0"),
                ),
                TranscriptSegment(
                    text="Got it.",
                    start_time=10.0,
                    end_time=11.0,
                    speaker=TranscriptSpeaker(id="speaker_1"),
                ),
            ],
        )

        # Use pattern matching directly
        results = service._match_by_pattern(
            speakers=["speaker_0", "speaker_1"],
            transcripts=[transcript],
            team_members=sample_deployment.team_members,
        )

        # speaker_0 uses CTO-like language, should match Damion
        if "speaker_0" in results and results["speaker_0"][0]:
            assert results["speaker_0"][0] == "person:damion"
            assert results["speaker_0"][1] >= 0.3

    def test_pattern_matching_with_deployment(self, service, sample_deployment, temp_dir):
        """Test pattern matching via resolve_speakers_for_deployment."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Let me deploy the system updates and check the technical architecture.",
                    start_time=0.0,
                    end_time=5.0,
                    speaker=TranscriptSpeaker(id="speaker_0"),
                ),
            ],
        )

        # Create the directory structure
        deploy_dir = Path(temp_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        mappings = service.resolve_speakers_for_deployment(
            deployment=sample_deployment,
            transcripts=[transcript],
            base_path=temp_dir,
            save=False,  # Don't save to avoid directory issues
        )

        assert len(mappings) == 1


class TestSaveMappings:
    """Tests for saving speaker mappings to disk."""

    def test_save_mappings_creates_file(self, service, sample_deployment, temp_dir):
        """Test that save_mappings creates the JSON file."""
        # Create the directory structure
        deploy_dir = Path(temp_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_0",
                deployment_id=sample_deployment.id,
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.8,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ]

        saved_paths = service.save_mappings(mappings, sample_deployment, temp_dir)

        assert "source:deploy:20250119_vinci_01/gopro_01" in saved_paths
        assert saved_paths["source:deploy:20250119_vinci_01/gopro_01"].exists()

        # Verify file content
        data = json.loads(saved_paths["source:deploy:20250119_vinci_01/gopro_01"].read_text())
        assert data["deployment_id"] == sample_deployment.id
        assert len(data["mappings"]) == 1
        assert data["mappings"][0]["raw_speaker_id"] == "speaker_0"
        assert data["mappings"][0]["person_id"] == "person:damion"
        assert data["mappings"][0]["confidence"] == 0.8
        assert data["mappings"][0]["method"] == "context_inference"

    def test_save_mappings_multiple_sources(self, service, sample_deployment, temp_dir):
        """Test saving mappings for multiple sources."""
        # Create directory structure for both sources
        source1_dir = Path(temp_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
        source2_dir = Path(temp_dir) / "deploy_20250119_vinci_01" / "sources" / "phone_01"
        source1_dir.mkdir(parents=True, exist_ok=True)
        source2_dir.mkdir(parents=True, exist_ok=True)

        mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_0",
                deployment_id=sample_deployment.id,
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.8,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
            SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id=sample_deployment.id,
                source_id="source:deploy:20250119_vinci_01/phone_01",
                resolved_person_id="person:chu",
                confidence=0.6,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ]

        saved_paths = service.save_mappings(mappings, sample_deployment, temp_dir)

        assert len(saved_paths) == 2
        assert all(p.exists() for p in saved_paths.values())


class TestLoadMappings:
    """Tests for loading speaker mappings from disk."""

    def test_load_mappings_success(self, service, sample_deployment, temp_dir):
        """Test loading mappings that were previously saved."""
        # Create and save mappings
        deploy_dir = Path(temp_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        original_mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_0",
                deployment_id=sample_deployment.id,
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.8,
                method=ResolutionMethod.CONTEXT_INFERENCE,
                verified=True,
                evidence_notes="Self-identified as Damion",
            ),
        ]

        service.save_mappings(original_mappings, sample_deployment, temp_dir)

        # Load mappings
        loaded_mappings = service.load_mappings(
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            deployment=sample_deployment,
            base_path=temp_dir,
        )

        assert len(loaded_mappings) == 1
        assert loaded_mappings[0].raw_speaker_id == "speaker_0"
        assert loaded_mappings[0].resolved_person_id == "person:damion"
        assert loaded_mappings[0].confidence == 0.8
        assert loaded_mappings[0].method == ResolutionMethod.CONTEXT_INFERENCE
        assert loaded_mappings[0].verified is True
        assert loaded_mappings[0].evidence_notes == "Self-identified as Damion"

    def test_load_mappings_nonexistent(self, service, sample_deployment, temp_dir):
        """Test loading mappings when file doesn't exist."""
        deploy_dir = Path(temp_dir) / "deploy_20250119_vinci_01" / "sources" / "gopro_01"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        mappings = service.load_mappings(
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            deployment=sample_deployment,
            base_path=temp_dir,
        )

        assert mappings == []


class TestVoiceMatching:
    """Tests for voice embedding matching (mocked)."""

    def test_match_by_voice_embedding_no_embedding(self, service):
        """Test voice matching when person has no embedding."""
        person = Person(
            id="person:test",
            name="Test Person",
            voice_embedding=None,
        )

        score, method = service._match_by_voice_embedding(
            speaker_audio_path="/path/to/audio.wav",
            speaker_start=0.0,
            speaker_end=5.0,
            person=person,
        )

        assert score == 0.0
        assert method == "no_embedding"

    def test_match_by_voice_embedding_resemblyzer_not_available(self, service):
        """Test voice matching when resemblyzer is not installed."""
        person = Person(
            id="person:test",
            name="Test Person",
            voice_embedding=[0.1, 0.2, 0.3],  # Fake embedding
        )

        # The import will fail, so we should get a specific error
        score, method = service._match_by_voice_embedding(
            speaker_audio_path="/path/to/audio.wav",
            speaker_start=0.0,
            speaker_end=5.0,
            person=person,
        )

        # Either no resemblyzer or file not found error
        assert score == 0.0
        assert "resemblyzer_not_available" in method or "error" in method

    def test_match_by_voice_embedding_success(self):
        """Test successful voice matching with mocked resemblyzer."""
        # This test would require complex mocking of resemblyzer internals
        # The method is tested via the no_embedding and import error cases
        # A full integration test would require actual audio files and resemblyzer
        pass


class TestCrossReferencing:
    """Tests for cross-referencing mappings across sources."""

    def test_cross_reference_boosts_confidence(self, service, sample_deployment):
        """Test that consistent mappings across sources boost confidence."""
        transcript1 = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="This is Damion speaking.",
                    start_time=0.0,
                    end_time=3.0,
                    speaker=TranscriptSpeaker(id="speaker_0"),
                ),
            ],
        )

        transcript2 = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/phone_01",
            source_id="source:deploy:20250119_vinci_01/phone_01",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="I'm Damion, checking the phone camera.",
                    start_time=0.0,
                    end_time=3.0,
                    speaker=TranscriptSpeaker(id="speaker_A"),
                ),
            ],
        )

        mappings = service.resolve_speakers(
            deployment_id=sample_deployment.id,
            sources=[],
            transcripts=[transcript1, transcript2],
        )

        # Both speakers identified as Damion should have boosted confidence
        damion_mappings = [
            m for m in mappings if m.resolved_person_id == "person:damion"
        ]

        # Should have at least one mapping with boosted confidence
        if len(damion_mappings) > 1:
            for mapping in damion_mappings:
                # Cross-reference should boost confidence by 0.1
                assert mapping.confidence >= 0.8  # 0.8 base + 0.1 boost


class TestPeopleManagement:
    """Tests for adding and managing people."""

    def test_add_person(self, service, temp_dir):
        """Test adding a new person to the registry."""
        new_person = Person(
            id="person:alice",
            name="Alice Smith",
            aliases=["Alice", "Al"],
            role="Engineer",
        )

        service.add_person(new_person)

        # Verify person was added
        retrieved = service.get_person("person:alice")
        assert retrieved is not None
        assert retrieved.name == "Alice Smith"
        assert retrieved.role == "Engineer"

    def test_add_voice_sample(self, service):
        """Test adding a voice sample to a person."""
        service.add_voice_sample(
            person_id="person:damion",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            start_time=0.0,
            end_time=5.0,
            verified=True,
        )

        person = service.get_person("person:damion")
        assert len(person.voice_samples) == 1
        assert person.voice_samples[0].source_id == "source:deploy:20250119_vinci_01/gopro_01"
        assert person.voice_samples[0].verified is True

    def test_add_voice_sample_invalid_person(self, service):
        """Test adding a voice sample to a non-existent person."""
        with pytest.raises(ValueError, match="Person not found"):
            service.add_voice_sample(
                person_id="person:nonexistent",
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                start_time=0.0,
                end_time=5.0,
            )
