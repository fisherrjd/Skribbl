Whisper Server for STT 

Pyannote - speaker enrollment / recognition

OBS recording everything 


OBS running → captures mic + desktop audio → saves to meeting.mkv
ffmpeg -i meeting.mkv -vn -acodec pcm_s16le -ar 16000 -ac 1 meeting.wav
Speaker diarization
Speaker Identification
Transcribe with whisper
Generate Notes



LLM PROMPT:
I'm building an automated meeting transcription and note-taking system. Here's what I have set up:

**Current Setup:**
- Windows machine for recording (OBS captures mic + desktop audio → MKV files)
- Linux server "eldo" running whisper.cpp server (whisper-server with ggml-small.en.bin model on port 5050)
- Whisper server accessible at http://eldo:5050/inference (accepts audio files via POST multipart/form-data with key "file")
- ffmpeg installed on Windows for audio extraction

**Goal:**
Build a pipeline that:
1. Takes OBS meeting recordings (MKV with stereo audio: my mic + others' voices)
2. Performs speaker diarization to identify who spoke when
3. Uses speaker enrollment to recognize specific people across multiple meetings
4. Transcribes each speaker segment with Whisper
5. Outputs formatted transcript with speaker names
6. (Future) Generate meeting notes with LLM

**Technology Stack I Want to Use:**
- Whisper.cpp for transcription (already running)
- pyannote.audio for speaker diarization and enrollment
- Python for orchestration
- ffmpeg for audio processing

**What I Need Help With:**
[Describe your specific next step - e.g., "Setting up pyannote.audio and creating the speaker enrollment system" or "Building the Python script to orchestrate the full pipeline"]

**Questions:**
- How do I structure the voice database for speaker enrollment?
- Should I process audio segments individually or batch them?
- What's the best way to handle poor audio quality or background noise?
