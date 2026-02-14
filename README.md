# Skribbl — Meeting Transcription Pipeline Plan

## Context

The goal is to take meeting recordings (MP4 from OBS) and produce speaker-labeled transcripts, all running locally on an M1 Mac with 16GB RAM. The current codebase has a skeleton diarization script using pyannote 3.1 and a whisper-cpp small English model, but nothing is wired together yet.

## Constraints

- **16GB RAM M1 Mac** — must run everything locally and sequentially (not simultaneously) to stay within memory
- **No external servers** — not using the eldo whisper server; everything runs on this machine
- **HuggingFace token required** — pyannote models (including the newer community-1) require accepting a license and providing an auth token

## Architecture: Sequential Pipeline

The pipeline runs in discrete, sequential stages to manage memory on 16GB:

```
MP4 → [ffmpeg] → WAV → [pyannote] → speaker segments → [whisper-cpp] → transcription → [merge] → labeled transcript
```

### Stage 1: Audio Extraction
- **Tool**: ffmpeg (already in nix env)
- **Action**: Extract audio from MP4 → 16kHz mono WAV
- **Command**: `ffmpeg -i recording.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav`
- 16kHz mono is the sweet spot — works for both whisper-cpp and pyannote

### Stage 2: Speaker Diarization
- **Tool**: pyannote.audio 4.0 with `speaker-diarization-community-1` model
- **Why community-1 over 3.1**: Better accuracy across all metrics, CC-BY-4.0 license, actively maintained. Still requires HF token but is fully open source.
- **Output**: List of `(start_time, end_time, speaker_label)` segments
- **Memory**: ~2-4GB for typical meeting-length audio (<1hr). Runs on CPU or MPS (Apple Silicon GPU). For longer recordings, memory can spike during clustering — we'll process on CPU to keep it manageable.
- **Key**: After diarization completes, we explicitly unload the model to free memory before whisper runs.

### Stage 3: Transcription
- **Tool**: whisper-cpp CLI (already in nix env) with `models/ggml-small.en.bin`
- **Approach**: Transcribe the **full audio file once** with word-level timestamps, rather than splitting into per-speaker segments
  - Why: Splitting audio loses context at boundaries, produces worse transcription quality, and is slower due to per-segment overhead
  - whisper-cpp outputs timestamps per segment that we can align with diarization
- **Output**: Timestamped transcript segments
- **Memory**: ~500MB for small.en model — very comfortable on 16GB

### Stage 4: Merge Diarization + Transcription
- **Pure Python, no ML models needed**
- Align whisper transcript segments with pyannote speaker segments by timestamp overlap
- For each whisper segment, assign the speaker who has the most overlap with that time range
- Output a clean transcript:
  ```
  [00:01:15] SPEAKER_00: So for this sprint, we need to focus on...
  [00:01:32] SPEAKER_01: Right, I think the priority should be...
  ```

### Stage 5 (Future): Speaker Identification / Enrollment
- **Not in initial implementation** — first version uses generic labels (SPEAKER_00, SPEAKER_01)
- Later: use pyannote's embedding model to extract voice embeddings from known speakers
- Store embeddings in a local voice database (JSON/SQLite + numpy arrays)
- At diarization time, match cluster centroids against enrolled embeddings to map SPEAKER_00 → "Alice"
- This is a natural extension once the base pipeline works

## File Structure

```
skribbl/
├── main.py                      # CLI entry point — orchestrates the pipeline
├── skribbl/
│   ├── __init__.py
│   ├── extract.py               # Stage 1: ffmpeg audio extraction
│   ├── diarize.py               # Stage 2: pyannote diarization
│   ├── transcribe.py            # Stage 3: whisper-cpp transcription
│   ├── merge.py                 # Stage 4: align and merge results
│   └── models.py                # Shared data types (Segment, Speaker, etc.)
├── diarize/                     # (existing — will be replaced by skribbl/diarize.py)
├── models/
│   └── ggml-small.en.bin        # Whisper model (already present)
├── recordings/                  # Input recordings
├── transcripts/                 # Output transcripts
└── pyproject.toml               # Add runtime deps: pyannote.audio, torch
```

## Dependencies to Add

In `pyproject.toml`:
```toml
dependencies = [
    "pyannote.audio>=3.3",
    "torch",
    "torchaudio",
]
```

whisper-cpp and ffmpeg are provided by the nix environment — no Python packages needed for those.

## CLI Interface

```bash
# Basic usage
python main.py recordings/Sprint_Planning_2_12_26.mp4

# Output goes to transcripts/Sprint_Planning_2_12_26.txt
```

Options:
- `--model` — whisper model path (default: `models/ggml-small.en.bin`)
- `--output` — output transcript path
- `--hf-token` — HuggingFace token (or read from `HF_TOKEN` env var)

## Implementation Order

1. **Data models** (`skribbl/models.py`) — define Segment, DiarizedSegment, TranscriptSegment types
2. **Audio extraction** (`skribbl/extract.py`) — ffmpeg wrapper, simple subprocess call
3. **Diarization** (`skribbl/diarize.py`) — pyannote wrapper, returns list of DiarizedSegments
4. **Transcription** (`skribbl/transcribe.py`) — whisper-cpp CLI wrapper, parses output into TranscriptSegments
5. **Merge** (`skribbl/merge.py`) — timestamp alignment logic
6. **CLI orchestration** (`main.py`) — wire it all together with argparse
7. **Test with the sprint planning recording**

## Verification

1. Run `ffmpeg` extraction on the MP4 and confirm a valid WAV is produced
2. Run diarization on the WAV and inspect speaker segments — do they look reasonable?
3. Run whisper-cpp on the WAV and confirm we get timestamped transcription
4. Run the merge and inspect the final labeled transcript
5. Compare against manually listening to sections of the recording to validate speaker labels



### COMMANDS

Hugging face Login
```
uvx hf auth login
```
