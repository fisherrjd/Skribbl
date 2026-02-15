# Quick Start Guide

Get started with Skribbl speaker identification in 3 steps.

## Prerequisites

```bash
# Install dependencies
uv sync

# Login to HuggingFace (required for pyannote)
uvx hf auth login
```

## Basic Usage

### 1. Enroll Speakers (One-Time Setup)

```bash
# Enroll team members with voice samples
python main.py enroll Alice recordings/alice_sample.wav
python main.py enroll Bob recordings/bob_sample.wav

# Verify enrollment
python main.py speakers
```

### 2. Transcribe Meetings

```bash
# Transcribe with automatic speaker identification
python main.py transcribe recordings/self.wav recordings/others.wav
```

### 3. Done!

Output will show names instead of SPEAKER_00, SPEAKER_01:
```
[00:01:15] Me: Let's start the meeting
[00:01:20] Alice: Sounds good
[00:01:35] Bob: I have an update
```

## Key Commands

```bash
# Transcribe a meeting
python main.py transcribe <self_audio> <others_audio>

# Enroll a speaker
python main.py enroll <name> <audio_file>

# List enrolled speakers
python main.py speakers

# Full help
python main.py help
```

## Voice Sample Requirements

- **Format:** WAV (16kHz mono preferred)
- **Duration:** 5-10 seconds of clear speech
- **Content:** Only ONE speaker
- **Quality:** Minimal background noise

### Standard Enrollment Script

Read this naturally (takes ~8-10 seconds):

> "Hello, my name is [YOUR NAME], and I'm recording this voice sample for speaker identification. I work on various projects involving software development, team collaboration, and technical discussions. This sample will help identify my voice in future meeting transcripts."

**See `ENROLLMENT_SCRIPT.md` for more scripts and recording tips.**

### How to Extract Voice Samples

From an existing meeting recording:

```bash
# Extract 8 seconds starting at timestamp 05:23
ffmpeg -i meeting.wav -ss 00:05:23 -t 00:00:08 alice_sample.wav
```

## File Structure

```
Skribbl/
├── main.py                    # Main CLI interface
├── diarize/
│   ├── transcribe.py         # Transcription with speaker ID
│   ├── voice_library.py      # Voice profile management
│   └── enroll.py             # Enrollment CLI
├── voice_profiles/           # Speaker profiles (auto-created)
│   ├── Alice.json           # Metadata
│   ├── Alice.npy            # Voice embedding
│   └── ...
├── recordings/               # Your audio files
└── transcripts/              # Output transcripts
```

## Next Steps

- **Detailed Guide:** See `VOICE_LIBRARY_GUIDE.md`
- **Example Workflow:** See `EXAMPLE_WORKFLOW.md`
- **Project Overview:** See `README.md`

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "No enrolled speakers" | Run `python main.py enroll <name> <audio>` |
| "Speaker not found" | Check `python main.py speakers` for enrolled names |
| "SPEAKER_00 instead of name" | Check audio quality or re-enroll with better sample |
| Module import errors | Run `uv sync` to install dependencies |

## Common Workflow

**First meeting with new team:**
1. Record → Transcribe (get generic labels)
2. Extract voice samples from the recording
3. Enroll everyone
4. Re-transcribe (optional) to get labeled output

**All future meetings:**
1. Record → Transcribe
2. Everyone automatically identified! ✨
