#!/usr/bin/env python
"""Skribbl - Meeting transcription with speaker identification."""

import sys
from pathlib import Path


def print_help():
    """Print help message."""
    print("""
Skribbl - Meeting Transcription with Speaker Identification

Usage:
    python main.py transcribe <self_audio> <others_audio>  - Transcribe a meeting
    python main.py enroll <name> <audio_file>              - Enroll a speaker
    python main.py speakers                                - List enrolled speakers
    python main.py help                                    - Show this help

Commands:

  transcribe - Transcribe and diarize meeting audio
    python main.py transcribe recordings/self.wav recordings/others.wav

    Outputs timestamped transcript with speaker labels. If speakers are enrolled,
    they will be identified by name. Otherwise uses generic labels (SPEAKER_00, etc.)

  enroll - Enroll a new speaker in the voice library
    python main.py enroll Alice recordings/alice_sample.wav

    The audio file should contain only one speaker (Alice) speaking for 5-10 seconds.
    This creates a voice profile that will be used to identify Alice in future meetings.

  speakers - List all enrolled speakers
    python main.py speakers

    Shows all speakers in the voice library with their enrollment details.

Examples:

  1. Transcribe a meeting (no speaker identification):
     python main.py transcribe recordings/self.wav recordings/others.wav

  2. Enroll team members:
     python main.py enroll Alice recordings/alice_intro.wav
     python main.py enroll Bob recordings/bob_intro.wav

  3. Transcribe meeting with speaker identification:
     python main.py transcribe recordings/self.wav recordings/others.wav

     Now the output will show:
     [00:01:15] Me: Let's start the meeting
     [00:01:20] Alice: Sounds good
     [00:01:25] Bob: I have an update

  4. Check enrolled speakers:
     python main.py speakers

Advanced:

  For more voice library management options, use the enroll.py script directly:
    python diarize/enroll.py list                    - List speakers
    python diarize/enroll.py info <name>             - Show speaker details
    python diarize/enroll.py delete <name>           - Delete a speaker
    python diarize/enroll.py enroll <name> <file>    - Enroll a speaker

Audio Requirements:

  - WAV format recommended
  - 16kHz sample rate (mono)
  - For enrollment: 5-10 seconds of clean speech (one speaker only)
  - For transcription: Use OBS to record separate mic and desktop audio tracks
""")


def cmd_transcribe():
    """Run transcription."""
    if len(sys.argv) < 4:
        print("Error: transcribe requires <self_audio> and <others_audio>")
        print("\nUsage: python main.py transcribe <self_audio> <others_audio>")
        print("\nExample:")
        print("  python main.py transcribe recordings/self.wav recordings/others.wav")
        sys.exit(1)

    self_audio = sys.argv[2]
    others_audio = sys.argv[3]

    # Import and run transcription
    sys.path.insert(0, str(Path(__file__).parent / "diarize"))
    from transcribe import main as transcribe_main

    # Replace sys.argv for transcribe.py
    sys.argv = ["transcribe.py", self_audio, others_audio]
    transcribe_main()


def cmd_enroll():
    """Enroll a speaker."""
    if len(sys.argv) < 4:
        print("Error: enroll requires <name> and <audio_file>")
        print("\nUsage: python main.py enroll <name> <audio_file>")
        print("\nExample:")
        print("  python main.py enroll Alice recordings/alice_sample.wav")
        sys.exit(1)

    name = sys.argv[2]
    audio_file = sys.argv[3]

    # Import and run enrollment
    sys.path.insert(0, str(Path(__file__).parent / "diarize"))
    from enroll import cmd_enroll as enroll_speaker
    from voice_library import VoiceLibrary

    library = VoiceLibrary()
    enroll_speaker(library, name, audio_file)


def cmd_speakers():
    """List enrolled speakers."""
    sys.path.insert(0, str(Path(__file__).parent / "diarize"))
    from enroll import cmd_list
    from voice_library import VoiceLibrary

    library = VoiceLibrary()
    cmd_list(library)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1]

    if command == "transcribe":
        cmd_transcribe()
    elif command == "enroll":
        cmd_enroll()
    elif command == "speakers":
        cmd_speakers()
    elif command in ["help", "-h", "--help"]:
        print_help()
    else:
        print(f"Error: Unknown command '{command}'")
        print("\nRun 'python main.py help' for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
