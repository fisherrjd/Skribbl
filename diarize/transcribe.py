import sys
from os import environ
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Inference, Pipeline
from voice_library import VoiceLibrary

_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": False}
)

load_dotenv()

HF_TOKEN = environ.get("hf_token", "")
DEVICE = "cpu"
WHISPER_MODEL = "large-v3"
BATCH_SIZE = 16
LANGUAGE = "en"


def transcribe_and_align(model, audio_path):
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)

    align_model, metadata = whisperx.load_align_model(
        language_code=LANGUAGE, device=DEVICE
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )
    return result, audio


def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def identify_speakers(
    diarize_result, others_audio: np.ndarray, voice_library: VoiceLibrary
) -> dict[str, str]:
    """
    Identify speakers by matching diarized segments to enrolled voice profiles.

    Args:
        diarize_result: Pyannote diarization result
        others_audio: Audio waveform (numpy array)
        voice_library: VoiceLibrary instance with enrolled speakers

    Returns:
        Dictionary mapping generic labels (SPEAKER_00) to identified names
    """
    if not voice_library.profiles:
        print("  No enrolled speakers - using generic labels", file=sys.stderr)
        return {}

    print("  Identifying speakers...", file=sys.stderr)

    # Get embedding model with auth token
    hf_token = environ.get("HF_TOKEN") or environ.get("hf_token")
    embedding_model = Inference(
        "pyannote/embedding", window="whole", use_auth_token=hf_token
    )

    # Extract speaker segments and compute embeddings
    speaker_embeddings = {}

    for segment, _, speaker_label in diarize_result.itertracks(yield_label=True):
        if speaker_label not in speaker_embeddings:
            # Extract audio segment for this speaker
            start_sample = int(segment.start * 16000)
            end_sample = int(segment.end * 16000)
            segment_audio = others_audio[start_sample:end_sample]

            # Compute embedding for this segment
            # Create temporary audio dict for pyannote
            audio_dict = {
                "waveform": torch.from_numpy(segment_audio).unsqueeze(0).float(),
                "sample_rate": 16000,
            }

            embedding = embedding_model(audio_dict)

            # Convert to numpy
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy().flatten()

            # Store first embedding for this speaker
            speaker_embeddings[speaker_label] = embedding

    # Match each speaker to enrolled profiles
    speaker_mapping = {}

    for speaker_label, embedding in speaker_embeddings.items():
        identified_name = voice_library.identify_speaker(embedding, threshold=0.7)

        if identified_name:
            speaker_mapping[speaker_label] = identified_name
            print(f"  ✓ {speaker_label} → {identified_name}", file=sys.stderr)
        else:
            print(f"  ? {speaker_label} → Unknown (no match found)", file=sys.stderr)

    return speaker_mapping


def main():
    self_path = sys.argv[1] if len(sys.argv) > 1 else "recordings/self.wav"
    others_path = sys.argv[2] if len(sys.argv) > 2 else "recordings/others.wav"

    if not Path(self_path).exists():
        print(f"Error: {self_path} not found", file=sys.stderr)
        sys.exit(1)
    if not Path(others_path).exists():
        print(f"Error: {others_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Loading WhisperX model...", file=sys.stderr)
    model = whisperx.load_model(
        WHISPER_MODEL, DEVICE, language=LANGUAGE, compute_type="int8"
    )

    # Transcribe self (single speaker, no diarization needed)
    print("Transcribing self...", file=sys.stderr)
    self_result, _ = transcribe_and_align(model, self_path)

    # Use enrolled name for self if available, otherwise "Me"
    self_speaker_name = "Me"
    voice_library = VoiceLibrary()
    voice_library.load_profiles()

    # Check if there's an enrolled speaker (you can specify via env var or arg)
    self_enrolled_name = environ.get("SELF_SPEAKER_NAME")
    if self_enrolled_name and self_enrolled_name in voice_library.profiles:
        self_speaker_name = self_enrolled_name

    for seg in self_result["segments"]:
        seg["speaker"] = self_speaker_name

    # Transcribe others (needs diarization)
    print("Transcribing others...", file=sys.stderr)
    others_result, others_audio = transcribe_and_align(model, others_path)

    print("Diarizing others...", file=sys.stderr)
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    )
    diarize_result = diarize_pipeline(
        {"waveform": torch.from_numpy(others_audio).unsqueeze(0), "sample_rate": 16000}
    )
    diarize_df = pd.DataFrame(
        diarize_result.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
    others_result = whisperx.assign_word_speakers(diarize_df, others_result)

    # Load voice library and identify speakers
    voice_library = VoiceLibrary()
    voice_library.load_profiles()

    speaker_mapping = {}
    if voice_library.profiles:
        speaker_mapping = identify_speakers(diarize_result, others_audio, voice_library)

    # Apply speaker mapping to segments
    for seg in others_result["segments"]:
        speaker_label = seg.get("speaker", "Unknown")
        # Map generic label to identified name if available
        if speaker_label in speaker_mapping:
            seg["speaker"] = speaker_mapping[speaker_label]

    # Merge both transcripts by timestamp
    all_segments = self_result["segments"] + others_result["segments"]
    all_segments.sort(key=lambda s: s["start"])

    # Output
    for seg in all_segments:
        ts = format_timestamp(seg["start"])
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        if text:
            print(f"[{ts}] {speaker}: {text}")


if __name__ == "__main__":
    main()
