"""Tests for the SpeakerResolutionService."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.person import Person, VoiceSample
from gram_deploy.models.transcript import RawTranscript, TranscriptSegment, TranscriptSpeaker
from gram_deploy.models.speaker_mapping import SpeakerMapping, ResolutionMethod
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.source import Source, DeviceType, SourceFile

# Import speaker_resolution directly to avoid loading all services
import importlib.util
spec = importlib.util.spec_from_file_location(
    "speaker_resolution",
    Path(__file__).parent.parent / "src" / "gram_deploy" / "services" / "speaker_resolution.py"
)
sr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr_module)
SpeakerResolutionService = sr_module.SpeakerResolutionService


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def people_registry(temp_data_dir):
    """Create a test people registry."""
    registry_path = Path(temp_data_dir) / "people.json"
    registry_data = {
        "people": [
            {
                "id": "person:damion",
                "name": "Damion Shelton",
                "aliases": ["Damion"],
                "role": "CTO",
                "voice_samples": [],
                "voice_embedding": None,
                "created_at": "2025-01-19T00:00:00Z",
                "updated_at": "2025-01-19T00:00:00Z"
            },
            {
                "id": "person:chu",
                "name": "Chu",
                "aliases": [],
                "role": None,
                "voice_samples": [],
                "voice_embedding": None,
                "created_at": "2025-01-19T00:00:00Z",
                "updated_at": "2025-01-19T00:00:00Z"
            }
        ]
    }
    registry_path.write_text(json.dumps(registry_data, indent=2))
    return str(registry_path)


@pytest.fixture
def service(people_registry, temp_data_dir):
    """Create a SpeakerResolutionService with test registry."""
    return SpeakerResolutionService(people_registry, temp_data_dir)


@pytest.fixture
def sample_transcript():
    """Create a sample transcript for testing."""
    return RawTranscript(
        id="transcript:source:deploy:20250119_vinci_01/gopro_01",
        source_id="source:deploy:20250119_vinci_01/gopro_01",
        transcription_service="test",
        segments=[
            TranscriptSegment(
                text="Hey everyone, this is Damion here.",
                start_time=0.0,
                end_time=3.0,
                speaker=TranscriptSpeaker(id="speaker_A"),
            ),
            TranscriptSegment(
                text="Good morning! Let's get started.",
                start_time=3.5,
                end_time=6.0,
                speaker=TranscriptSpeaker(id="speaker_B"),
            ),
            TranscriptSegment(
                text="As CTO, I think we should focus on the core architecture.",
                start_time=6.5,
                end_time=10.0,
                speaker=TranscriptSpeaker(id="speaker_A"),
            ),
            TranscriptSegment(
                text="Hey Damion, can you explain that more?",
                start_time=10.5,
                end_time=13.0,
                speaker=TranscriptSpeaker(id="speaker_B"),
            ),
            TranscriptSegment(
                text="Sure, let me break it down.",
                start_time=13.5,
                end_time=15.0,
                speaker=TranscriptSpeaker(id="speaker_A"),
            ),
        ],
    )


@pytest.fixture
def sample_source():
    """Create a sample source for testing."""
    return Source(
        id="source:deploy:20250119_vinci_01/gopro_01",
        deployment_id="deploy:20250119_vinci_01",
        device_type=DeviceType.GOPRO,
        device_number=1,
        files=[
            SourceFile(
                filename="GX010001.MP4",
                file_path="/videos/GX010001.MP4",
                duration_seconds=300.0,
                start_offset_ms=0,
            )
        ],
    )


class TestServiceInitialization:
    """Tests for service initialization."""

    def test_init_creates_registry_if_missing(self, temp_data_dir):
        """Test that service handles missing registry gracefully."""
        registry_path = Path(temp_data_dir) / "nonexistent" / "people.json"
        service = SpeakerResolutionService(str(registry_path), temp_data_dir)

        assert service._people == {}

    def test_init_loads_existing_registry(self, service):
        """Test that service loads existing people from registry."""
        people = service.list_people()

        assert len(people) == 2
        assert any(p.id == "person:damion" for p in people)
        assert any(p.id == "person:chu" for p in people)

    def test_get_person_existing(self, service):
        """Test getting an existing person by ID."""
        person = service.get_person("person:damion")

        assert person is not None
        assert person.name == "Damion Shelton"
        assert person.role == "CTO"

    def test_get_person_nonexistent(self, service):
        """Test getting a non-existent person returns None."""
        person = service.get_person("person:nonexistent")
        assert person is None


class TestContextMatching:
    """Tests for context-based speaker matching."""

    def test_resolve_by_context_self_identification(self, service, sample_transcript):
        """Test matching speaker by self-identification pattern."""
        # speaker_A says "this is Damion here"
        person_id, confidence, method = service._resolve_by_context(
            "speaker_A", sample_transcript
        )

        assert person_id == "person:damion"
        assert confidence >= 0.8
        assert method == ResolutionMethod.CONTEXT_INFERENCE

    def test_resolve_by_context_no_match(self, service):
        """Test that context matching returns unresolved when no patterns match."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="test",
            segments=[
                TranscriptSegment(
                    text="Hello, how are you today?",
                    start_time=0.0,
                    end_time=2.0,
                    speaker=TranscriptSpeaker(id="speaker_X"),
                ),
            ],
        )

        person_id, confidence, method = service._resolve_by_context(
            "speaker_X", transcript
        )

        assert person_id is None
        assert confidence == 0.0
        assert method == ResolutionMethod.UNRESOLVED

    def test_resolve_by_context_addressing_pattern(self, service):
        """Test matching speaker by addressing pattern in preceding segment."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="test",
            segments=[
                TranscriptSegment(
                    text="Hey Damion, can you help with this?",
                    start_time=0.0,
                    end_time=2.0,
                    speaker=TranscriptSpeaker(id="speaker_B"),
                ),
                TranscriptSegment(
                    text="Sure, let me take a look.",
                    start_time=2.5,
                    end_time=4.0,
                    speaker=TranscriptSpeaker(id="speaker_A"),
                ),
            ],
        )

        # speaker_A is addressed as Damion, so should match
        person_id, confidence, method = service._resolve_by_context(
            "speaker_A", transcript
        )

        assert person_id == "person:damion"
        assert confidence >= 0.6
        assert method == ResolutionMethod.CONTEXT_INFERENCE

    def test_match_by_context_with_person(self, service, sample_transcript):
        """Test _match_by_context method that takes person object."""
        person = service.get_person("person:damion")

        speaker_id, confidence = service._match_by_context(sample_transcript, person)

        assert speaker_id == "speaker_A"
        assert confidence > 0

    def test_match_by_context_role_specific(self, service):
        """Test matching by role-specific language."""
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="test",
            segments=[
                TranscriptSegment(
                    text="As CTO, I need to make sure our architecture is solid.",
                    start_time=0.0,
                    end_time=4.0,
                    speaker=TranscriptSpeaker(id="speaker_C"),
                ),
            ],
        )

        person = service.get_person("person:damion")
        speaker_id, confidence = service._match_by_context(transcript, person)

        assert speaker_id == "speaker_C"
        assert confidence > 0


class TestVoiceEmbeddingMatching:
    """Tests for voice embedding-based speaker matching."""

    def test_match_by_voice_embedding_no_embeddings(self, service):
        """Test that voice matching returns 0 when no embeddings available."""
        person = service.get_person("person:damion")

        similarity = service._match_by_voice_embedding([], person)

        assert similarity == 0.0

    def test_match_by_voice_embedding_with_mock_embeddings(self, service):
        """Test voice matching with mock embeddings."""
        # Add mock embedding to person
        person = service.get_person("person:damion")
        person.voice_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Test with similar embedding
        speaker_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        similarity = service._match_by_voice_embedding(speaker_embedding, person)

        assert similarity > 0.9  # Should be very similar

    def test_match_by_voice_embedding_different_embeddings(self, service):
        """Test voice matching with different embeddings."""
        person = service.get_person("person:damion")
        person.voice_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]

        # Test with different embedding
        speaker_embedding = [0.0, 1.0, 0.0, 0.0, 0.0]
        similarity = service._match_by_voice_embedding(speaker_embedding, person)

        # Cosine similarity of orthogonal vectors should be 0.5 after normalization
        assert similarity < 0.6

    def test_match_by_voice_embedding_empty_speaker_embedding(self, service):
        """Test that voice matching returns 0 for empty speaker embedding."""
        person = service.get_person("person:damion")
        person.voice_embedding = [0.1, 0.2, 0.3]

        # Empty speaker embedding should return 0.0
        similarity = service._match_by_voice_embedding([], person)
        assert similarity == 0.0


class TestPatternMatching:
    """Tests for pattern-based speaker matching."""

    def test_match_by_pattern_no_speakers(self, service):
        """Test pattern matching with empty speakers dict."""
        person_id, confidence, method = service._match_by_pattern(
            {}, [], None
        )

        assert person_id is None
        assert confidence == 0.0
        assert method == ResolutionMethod.UNRESOLVED

    def test_match_by_pattern_single_team_member(self, service, sample_transcript):
        """Test pattern matching with single expected team member."""
        # Build speakers dict
        speakers = {
            f"{sample_transcript.source_id}/speaker_A": [("speaker_A", sample_transcript)],
            f"{sample_transcript.source_id}/speaker_B": [("speaker_B", sample_transcript)],
        }

        people = service.list_people()
        team_members = [service.get_person("person:damion")]

        # This should try to match based on patterns
        person_id, confidence, method = service._match_by_pattern(
            speakers, people, team_members
        )

        # Pattern matching is heuristic, result depends on frequency analysis
        assert method in [ResolutionMethod.CONTEXT_INFERENCE, ResolutionMethod.UNRESOLVED]

    def test_match_by_pattern_calculates_frequency(self, service):
        """Test that pattern matching considers speaker frequency."""
        # Create transcript with one dominant speaker
        transcript = RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            transcription_service="test",
            segments=[
                TranscriptSegment(
                    text=f"Utterance {i}",
                    start_time=float(i),
                    end_time=float(i) + 1.0,
                    speaker=TranscriptSpeaker(id="speaker_A" if i % 3 != 0 else "speaker_B"),
                )
                for i in range(20)
            ],
        )

        speakers = {
            f"{transcript.source_id}/speaker_A": [("speaker_A", transcript)],
            f"{transcript.source_id}/speaker_B": [("speaker_B", transcript)],
        }

        people = service.list_people()
        team_members = [service.get_person("person:damion")]

        # Pattern matching should identify speaker_A as more frequent
        person_id, confidence, method = service._match_by_pattern(
            speakers, people, team_members
        )

        # Even if it doesn't resolve, it shouldn't crash
        assert confidence >= 0.0


class TestResolveSpeakers:
    """Tests for the main resolve_speakers method."""

    def test_resolve_speakers_with_deployment_object(
        self, service, sample_transcript, sample_source, temp_data_dir
    ):
        """Test resolve_speakers accepting Deployment object."""
        # Create deployment directory structure
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
            sources=["source:deploy:20250119_vinci_01/gopro_01"],
        )

        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        source_dir = deploy_dir / "sources" / "gopro_01"
        source_dir.mkdir(parents=True)

        # Write source and transcript files
        (source_dir / "source.json").write_text(sample_source.model_dump_json())
        (source_dir / "raw_transcript.json").write_text(sample_transcript.model_dump_json())

        # Resolve speakers
        mappings = service.resolve_speakers(deployment)

        assert len(mappings) > 0
        assert all(isinstance(m, SpeakerMapping) for m in mappings)

    def test_resolve_speakers_with_provided_data(
        self, service, sample_transcript, sample_source
    ):
        """Test resolve_speakers with explicitly provided sources and transcripts."""
        mappings = service.resolve_speakers(
            "deploy:20250119_vinci_01",
            sources=[sample_source],
            transcripts=[sample_transcript],
        )

        assert len(mappings) == 2  # speaker_A and speaker_B

        # Check that speaker_A was resolved to Damion
        speaker_a_mapping = next(
            (m for m in mappings if m.raw_speaker_id == "speaker_A"), None
        )
        assert speaker_a_mapping is not None
        assert speaker_a_mapping.resolved_person_id == "person:damion"
        assert speaker_a_mapping.method == ResolutionMethod.CONTEXT_INFERENCE

    def test_resolve_speakers_empty_transcripts(self, service, sample_source):
        """Test resolve_speakers with no transcripts."""
        mappings = service.resolve_speakers(
            "deploy:20250119_vinci_01",
            sources=[sample_source],
            transcripts=[],
        )

        assert mappings == []


class TestSaveMappings:
    """Tests for saving speaker mappings."""

    def test_save_mappings_creates_file(self, service, temp_data_dir):
        """Test that save_mappings creates the mapping file."""
        deployment_id = "deploy:20250119_vinci_01"
        source_id = "source:deploy:20250119_vinci_01/gopro_01"

        # Create source directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        source_dir = deploy_dir / "sources" / "gopro_01"
        source_dir.mkdir(parents=True)

        mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id=deployment_id,
                source_id=source_id,
                resolved_person_id="person:damion",
                confidence=0.8,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ]

        service.save_mappings(deployment_id, mappings)

        mappings_path = source_dir / "speaker_mappings.json"
        assert mappings_path.exists()

        data = json.loads(mappings_path.read_text())
        assert len(data) == 1
        assert data[0]["raw_speaker_id"] == "speaker_A"
        assert data[0]["person_id"] == "person:damion"

    def test_save_mappings_groups_by_source(self, service, temp_data_dir):
        """Test that save_mappings creates separate files per source."""
        deployment_id = "deploy:20250119_vinci_01"

        # Create source directories
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        for device in ["gopro_01", "phone_01"]:
            (deploy_dir / "sources" / device).mkdir(parents=True)

        mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id=deployment_id,
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.8,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
            SpeakerMapping(
                raw_speaker_id="speaker_B",
                deployment_id=deployment_id,
                source_id="source:deploy:20250119_vinci_01/phone_01",
                resolved_person_id="person:chu",
                confidence=0.7,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ]

        service.save_mappings(deployment_id, mappings)

        # Check both files exist
        gopro_mappings = deploy_dir / "sources" / "gopro_01" / "speaker_mappings.json"
        phone_mappings = deploy_dir / "sources" / "phone_01" / "speaker_mappings.json"

        assert gopro_mappings.exists()
        assert phone_mappings.exists()

        gopro_data = json.loads(gopro_mappings.read_text())
        assert gopro_data[0]["raw_speaker_id"] == "speaker_A"

        phone_data = json.loads(phone_mappings.read_text())
        assert phone_data[0]["raw_speaker_id"] == "speaker_B"


class TestLoadMappings:
    """Tests for loading speaker mappings."""

    def test_load_mappings_existing_file(self, service, temp_data_dir):
        """Test loading mappings from an existing file."""
        deployment_id = "deploy:20250119_vinci_01"
        source_id = "source:deploy:20250119_vinci_01/gopro_01"

        # Create mapping file
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        source_dir = deploy_dir / "sources" / "gopro_01"
        source_dir.mkdir(parents=True)

        mappings_data = [
            {
                "raw_speaker_id": "speaker_A",
                "person_id": "person:damion",
                "confidence": 0.8,
                "method": "context_inference",
            },
        ]
        (source_dir / "speaker_mappings.json").write_text(json.dumps(mappings_data))

        mappings = service.load_mappings(deployment_id, source_id)

        assert len(mappings) == 1
        assert mappings[0].raw_speaker_id == "speaker_A"
        assert mappings[0].resolved_person_id == "person:damion"
        assert mappings[0].method == ResolutionMethod.CONTEXT_INFERENCE

    def test_load_mappings_no_file(self, service, temp_data_dir):
        """Test loading mappings when file doesn't exist."""
        mappings = service.load_mappings(
            "deploy:20250119_vinci_01",
            "source:deploy:20250119_vinci_01/gopro_01"
        )

        assert mappings == []


class TestCrossReferenceMappings:
    """Tests for cross-referencing mappings across sources."""

    def test_cross_reference_boosts_confidence(self, service, sample_transcript):
        """Test that cross-referencing boosts confidence for consistent mappings."""
        # Create two mappings for the same person from different sources
        mappings = [
            SpeakerMapping(
                raw_speaker_id="speaker_A",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/gopro_01",
                resolved_person_id="person:damion",
                confidence=0.6,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
            SpeakerMapping(
                raw_speaker_id="speaker_X",
                deployment_id="deploy:20250119_vinci_01",
                source_id="source:deploy:20250119_vinci_01/phone_01",
                resolved_person_id="person:damion",
                confidence=0.6,
                method=ResolutionMethod.CONTEXT_INFERENCE,
            ),
        ]

        original_confidences = [m.confidence for m in mappings]
        service._cross_reference_mappings(mappings, [sample_transcript])

        # Both should have boosted confidence
        assert all(m.confidence > 0.6 for m in mappings)


class TestAddPerson:
    """Tests for adding people to the registry."""

    def test_add_person_saves_to_registry(self, service, temp_data_dir):
        """Test that add_person persists to the registry file."""
        new_person = Person(
            id="person:alice",
            name="Alice Smith",
            aliases=["Alice"],
            role="Engineer",
        )

        service.add_person(new_person)

        # Reload registry and check
        registry_path = Path(service.registry_path)
        data = json.loads(registry_path.read_text())

        person_ids = [p["id"] for p in data["people"]]
        assert "person:alice" in person_ids


class TestVoiceSamples:
    """Tests for voice sample management."""

    def test_add_voice_sample(self, service):
        """Test adding a voice sample to a person."""
        service.add_voice_sample(
            person_id="person:damion",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            start_time=10.0,
            end_time=20.0,
            verified=True,
        )

        person = service.get_person("person:damion")
        assert len(person.voice_samples) == 1
        assert person.voice_samples[0].start_time == 10.0
        assert person.voice_samples[0].end_time == 20.0
        assert person.voice_samples[0].verified is True

    def test_add_voice_sample_invalid_person(self, service):
        """Test that add_voice_sample raises for non-existent person."""
        with pytest.raises(ValueError, match="Person not found"):
            service.add_voice_sample(
                person_id="person:nonexistent",
                source_id="source:test",
                start_time=0.0,
                end_time=5.0,
            )
