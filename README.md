# Skribbl — Meeting Transcription with Speaker Identification

Local meeting transcription pipeline for M1 Mac that produces speaker-labeled transcripts from audio recordings.

## Current Status

**Working:**
- Speaker diarization using pyannote.audio 3.1
- Full meeting transcription with WhisperX
- Timestamped speaker-labeled output

**In Progress:**
- Voice library system for speaker identification (enrolling known speakers)
- Converting generic labels (SPEAKER_00, SPEAKER_01) to actual names

## Project Structure

```
Skribbl/
├── main.py                          # Entry point (currently minimal)
├── diarize/
│   ├── diarize_meeting.py          # Standalone pyannote diarization (2 speakers)
│   └── transcribe.py               # Full pipeline: WhisperX + pyannote + alignment
├── voice_profiles/                  # Storage for enrolled speaker voice embeddings
├── recordings/                      # Input audio files (WAV)
├── transcripts/                     # Output transcripts
│   └── Peackock_Meeting.md         # Example output
├── models/                          # Whisper models
└── pyproject.toml                   # Dependencies

```

## Current Pipeline

### diarize/transcribe.py

The main transcription script that handles everything end-to-end:

1. **Transcribe "self" track** (your microphone) with WhisperX
   - Labels all segments as "Me"
   - Uses alignment model for word-level timestamps

2. **Transcribe "others" track** (desktop audio/other participants) with WhisperX
   - Full transcription with alignment

3. **Diarize "others" track** with pyannote
   - Identifies different speakers in the mixed audio
   - Creates speaker segments

4. **Assign speakers to transcription**
   - Maps diarized speaker segments to transcribed words
   - Labels speakers as SPEAKER_00, SPEAKER_01, etc.

5. **Merge and output**
   - Combines self + others transcripts
   - Sorts by timestamp
   - Outputs formatted transcript

**Usage:**
```bash
python diarize/transcribe.py recordings/self.wav recordings/others.wav
```

**Output format:**
```
[00:01:15] Me: So for this sprint, we need to focus on...
[00:01:32] SPEAKER_00: Right, I think the priority should be...
[00:01:45] SPEAKER_01: I agree with that approach.
```

### diarize/diarize_meeting.py

Standalone speaker diarization script (simpler version for testing):

```bash
python diarize/diarize_meeting.py system.wav
```

Outputs speaker segments:
```
15.2s - 28.4s: SPEAKER_00
28.6s - 42.1s: SPEAKER_01
```

## Technology Stack

- **WhisperX**: Speech-to-text with word-level alignment
  - Model: large-v3
  - Language: English
  - Device: CPU (with int8 quantization for M1 efficiency)

- **pyannote.audio 3.1**: Speaker diarization and voice embeddings
  - Requires HuggingFace token (see setup)

- **Python 3.13**: Runtime

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. HuggingFace Authentication

pyannote models require accepting their license and providing a token:

1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the license
3. Create an access token at https://huggingface.co/settings/tokens
4. Login with token:

```bash
uvx hf auth login
```

Or set environment variable:
```bash
export HF_TOKEN="your_token_here"
```

Add to `.env` file:
```
hf_token=your_token_here
```

### 3. Prepare audio files

If you have separate mic tracks from OBS:
- `recordings/self.wav` - Your microphone
- `recordings/others.wav` - Desktop audio/other participants

Both should be 16kHz mono WAV files.

## Planned: Voice Library System

**Goal:** Replace generic speaker labels with actual names

**Approach:**
1. **Enrollment**: Extract voice embeddings from known speaker samples
2. **Storage**: Save embeddings in `voice_profiles/` with speaker metadata
3. **Identification**: Match diarized clusters against enrolled speakers
4. **Labeling**: Map SPEAKER_XX → "Alice", "Bob", etc.

**Implementation:**
- Use pyannote's embedding model to extract voice signatures
- Store embeddings as numpy arrays with JSON metadata
- Cosine similarity matching during diarization
- CLI tools for enrolling/managing speaker profiles

## Dependencies

Key packages (see `pyproject.toml` for full list):
- `whisperx>=3.7.6` - Transcription with alignment
- `python-dotenv>=1.2.1` - Environment config
- pyannote.audio (transitively via whisperx)
- torch, torchaudio (for ML models)

Dev tools:
- black, ruff (formatting/linting)
- pytest, pytest-cov (testing)

## Constraints

- **16GB RAM M1 Mac** - All processing runs locally
- **No external servers** - Everything on this machine
- **Sequential processing** - To manage memory, don't run transcription + diarization simultaneously

## Examples

See `transcripts/Peackock_Meeting.md` for example output from a real meeting recording.

## Documentation

📚 **[Complete Documentation Index](DOCUMENTATION_INDEX.md)** - Your guide to all documentation

**Quick Links:**
- **[Quick Start Guide](QUICK_START.md)** - Get started in 3 steps
- **[Enrollment Script](ENROLLMENT_SCRIPT.md)** - Standard phrases for voice enrollment
- **[Voice Library Guide](VOICE_LIBRARY_GUIDE.md)** - Complete reference for speaker identification
- **[Example Workflow](EXAMPLE_WORKFLOW.md)** - Real-world usage walkthrough
- **[Architecture](ARCHITECTURE.md)** - Technical deep dive

## Current Status

### ✅ Completed
- WhisperX transcription with word-level alignment
- Pyannote speaker diarization
- Voice library system for speaker enrollment
- Speaker identification (maps SPEAKER_00 → names)
- Unified CLI interface (`main.py`)
- Comprehensive documentation

### 🚀 Ready to Use
The system is fully functional! You can:
1. Enroll team members with voice samples
2. Transcribe meetings with automatic speaker identification
3. Get labeled transcripts with actual names

### 🔮 Future Enhancements
- Confidence scores for speaker matches
- Multi-sample enrollment (average multiple samples)
- Web UI for voice library management
- LLM-based meeting notes generation
