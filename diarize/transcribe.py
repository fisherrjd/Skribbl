import sys
from os import environ
from pathlib import Path

import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline

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

    for seg in self_result["segments"]:
        seg["speaker"] = "Me"

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
