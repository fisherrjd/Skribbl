"""Voice library for speaker enrollment and identification."""

import json
from dataclasses import asdict, dataclass
from os import environ
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Inference

# Workaround for pyannote model loading issue with PyTorch 2.6+
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": False}
)

load_dotenv()


@dataclass
class SpeakerProfile:
    """Represents an enrolled speaker with their voice embedding."""

    name: str
    embedding: np.ndarray
    enrollment_file: str
    created_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (without embedding)."""
        return {
            "name": self.name,
            "enrollment_file": self.enrollment_file,
            "created_at": self.created_at,
        }


class VoiceLibrary:
    """Manages speaker profiles and voice embeddings."""

    def __init__(self, profiles_dir: str | Path = "voice_profiles"):
        """
        Initialize voice library.

        Args:
            profiles_dir: Directory to store speaker profiles
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True, parents=True)
        self.profiles: dict[str, SpeakerProfile] = {}
        self._embedding_model: Optional[Inference] = None

    def _get_embedding_model(self) -> Inference:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            print("Loading pyannote embedding model...")
            hf_token = environ.get("HF_TOKEN") or environ.get("hf_token")
            self._embedding_model = Inference(
                "pyannote/embedding",
                window="whole",
                use_auth_token=hf_token,
            )
        return self._embedding_model

    def enroll_speaker(
        self,
        name: str,
        audio_file: str | Path,
        overwrite: bool = False,
    ) -> SpeakerProfile:
        """
        Enroll a new speaker from an audio sample.

        Args:
            name: Speaker name (used as identifier)
            audio_file: Path to audio file containing only this speaker
            overwrite: If True, overwrite existing profile with same name

        Returns:
            SpeakerProfile with extracted embedding

        Raises:
            ValueError: If speaker already enrolled and overwrite=False
        """
        audio_file = Path(audio_file)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Check if already enrolled
        if self.speaker_exists(name) and not overwrite:
            raise ValueError(
                f"Speaker '{name}' already enrolled. Use overwrite=True to replace."
            )

        # Extract embedding
        print(f"Extracting voice embedding for {name}...")
        model = self._get_embedding_model()
        embedding = model(str(audio_file))

        # Convert to numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        # Create profile
        from datetime import datetime

        profile = SpeakerProfile(
            name=name,
            embedding=embedding,
            enrollment_file=str(audio_file),
            created_at=datetime.now().isoformat(),
        )

        # Save to disk
        self._save_profile(profile)
        self.profiles[name] = profile

        print(f"✓ Enrolled speaker: {name}")
        return profile

    def _save_profile(self, profile: SpeakerProfile):
        """Save speaker profile to disk."""
        # Save embedding as numpy file
        embedding_path = self.profiles_dir / f"{profile.name}.npy"
        np.save(embedding_path, profile.embedding)

        # Save metadata as JSON
        metadata_path = self.profiles_dir / f"{profile.name}.json"
        with open(metadata_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def load_profiles(self):
        """Load all speaker profiles from disk."""
        self.profiles.clear()

        for metadata_file in self.profiles_dir.glob("*.json"):
            name = metadata_file.stem

            # Load metadata
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Load embedding
            embedding_path = self.profiles_dir / f"{name}.npy"
            if not embedding_path.exists():
                print(f"Warning: Missing embedding file for {name}, skipping")
                continue

            embedding = np.load(embedding_path)

            # Create profile
            profile = SpeakerProfile(
                name=metadata["name"],
                embedding=embedding,
                enrollment_file=metadata["enrollment_file"],
                created_at=metadata["created_at"],
            )

            self.profiles[name] = profile

        print(f"Loaded {len(self.profiles)} speaker profile(s)")

    def speaker_exists(self, name: str) -> bool:
        """Check if a speaker is enrolled."""
        metadata_path = self.profiles_dir / f"{name}.json"
        return metadata_path.exists()

    def identify_speaker(
        self,
        embedding: np.ndarray,
        threshold: float = 0.7,
    ) -> Optional[str]:
        """
        Identify a speaker by comparing embedding to enrolled speakers.

        Args:
            embedding: Voice embedding to identify
            threshold: Minimum cosine similarity to consider a match (0-1)

        Returns:
            Speaker name if match found, None otherwise
        """
        if not self.profiles:
            return None

        best_match = None
        best_score = threshold

        for name, profile in self.profiles.items():
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, profile.embedding)

            if similarity > best_score:
                best_score = similarity
                best_match = name

        return best_match

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def list_speakers(self) -> list[str]:
        """Get list of all enrolled speaker names."""
        return list(self.profiles.keys())

    def get_profile(self, name: str) -> Optional[SpeakerProfile]:
        """Get a speaker profile by name."""
        return self.profiles.get(name)

    def delete_speaker(self, name: str) -> bool:
        """
        Delete a speaker profile.

        Args:
            name: Speaker name to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self.profiles:
            return False

        # Remove from memory
        del self.profiles[name]

        # Remove from disk
        embedding_path = self.profiles_dir / f"{name}.npy"
        metadata_path = self.profiles_dir / f"{name}.json"

        embedding_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

        print(f"✓ Deleted speaker: {name}")
        return True
