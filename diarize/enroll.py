#!/usr/bin/env python
"""CLI tool for enrolling and managing speaker profiles."""

import sys
from pathlib import Path

from voice_library import VoiceLibrary


def print_usage():
    """Print usage information."""
    print("""
Skribbl Voice Library Management

Usage:
    python diarize/enroll.py enroll <name> <audio_file>   - Enroll a new speaker
    python diarize/enroll.py list                         - List all enrolled speakers
    python diarize/enroll.py info <name>                  - Show speaker profile info
    python diarize/enroll.py delete <name>                - Delete a speaker profile

Examples:
    # Enroll Alice from her voice sample
    python diarize/enroll.py enroll Alice recordings/alice_sample.wav

    # List all enrolled speakers
    python diarize/enroll.py list

    # Show Alice's profile info
    python diarize/enroll.py info Alice

    # Delete Bob's profile
    python diarize/enroll.py delete Bob

Notes:
    - Audio files should contain only one speaker
    - Use 5-10 seconds of clean speech for best results
    - WAV format recommended, 16kHz mono preferred
""")


def cmd_enroll(library: VoiceLibrary, name: str, audio_file: str):
    """Enroll a new speaker."""
    audio_path = Path(audio_file)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    # Check if overwrite needed
    overwrite = False
    if library.speaker_exists(name):
        response = input(
            f"Speaker '{name}' already enrolled. Overwrite? (y/N): "
        ).lower()
        if response != "y":
            print("Enrollment cancelled.")
            sys.exit(0)
        overwrite = True

    try:
        profile = library.enroll_speaker(name, audio_path, overwrite=overwrite)
        print(f"\n✓ Successfully enrolled: {profile.name}")
        print(f"  Audio file: {profile.enrollment_file}")
        print(f"  Enrolled at: {profile.created_at}")
        print(f"  Embedding shape: {profile.embedding.shape}")
    except Exception as e:
        print(f"Error enrolling speaker: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(library: VoiceLibrary):
    """List all enrolled speakers."""
    library.load_profiles()
    speakers = library.list_speakers()

    if not speakers:
        print("No speakers enrolled yet.")
        print("\nTo enroll a speaker:")
        print("  python diarize/enroll.py enroll <name> <audio_file>")
        return

    print(f"\nEnrolled speakers ({len(speakers)}):")
    print("-" * 60)

    for name in sorted(speakers):
        profile = library.get_profile(name)
        if profile:
            print(f"  {profile.name}")
            print(f"    File: {profile.enrollment_file}")
            print(f"    Enrolled: {profile.created_at}")
            print()


def cmd_info(library: VoiceLibrary, name: str):
    """Show detailed info about a speaker."""
    library.load_profiles()
    profile = library.get_profile(name)

    if not profile:
        print(f"Error: Speaker '{name}' not found.", file=sys.stderr)
        print("\nEnrolled speakers:")
        for speaker in library.list_speakers():
            print(f"  - {speaker}")
        sys.exit(1)

    print(f"\nSpeaker Profile: {profile.name}")
    print("-" * 60)
    print(f"Name:              {profile.name}")
    print(f"Enrollment file:   {profile.enrollment_file}")
    print(f"Enrolled at:       {profile.created_at}")
    print(f"Embedding shape:   {profile.embedding.shape}")
    print(f"Embedding mean:    {profile.embedding.mean():.4f}")
    print(f"Embedding std:     {profile.embedding.std():.4f}")


def cmd_delete(library: VoiceLibrary, name: str):
    """Delete a speaker profile."""
    library.load_profiles()

    if not library.speaker_exists(name):
        print(f"Error: Speaker '{name}' not found.", file=sys.stderr)
        sys.exit(1)

    response = input(f"Delete speaker '{name}'? This cannot be undone. (y/N): ").lower()
    if response != "y":
        print("Deletion cancelled.")
        sys.exit(0)

    if library.delete_speaker(name):
        print(f"✓ Deleted speaker: {name}")
    else:
        print(f"Error deleting speaker: {name}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]
    library = VoiceLibrary()

    if command == "enroll":
        if len(sys.argv) != 4:
            print("Error: enroll requires <name> and <audio_file>", file=sys.stderr)
            print("\nUsage: python diarize/enroll.py enroll <name> <audio_file>")
            sys.exit(1)
        cmd_enroll(library, sys.argv[2], sys.argv[3])

    elif command == "list":
        cmd_list(library)

    elif command == "info":
        if len(sys.argv) != 3:
            print("Error: info requires <name>", file=sys.stderr)
            print("\nUsage: python diarize/enroll.py info <name>")
            sys.exit(1)
        cmd_info(library, sys.argv[2])

    elif command == "delete":
        if len(sys.argv) != 3:
            print("Error: delete requires <name>", file=sys.stderr)
            print("\nUsage: python diarize/enroll.py delete <name>")
            sys.exit(1)
        cmd_delete(library, sys.argv[2])

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
