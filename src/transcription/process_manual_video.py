#!/usr/bin/env python3
"""
Process manually downloaded YouTube videos with Whisper
"""

import os
import sys
from pathlib import Path
import whisper
from .audio_transcription_pipeline import AudioTranscriptionPipeline

def find_video_files():
    """Find video/audio files in the project directory"""
    media_extensions = ['.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.m4a', '.mp3', '.wav']
    media_files = []

    # Search in current directory and subdirectories
    for ext in media_extensions:
        for file_path in Path('.').rglob(f'*{ext}'):
            if file_path.is_file():
                media_files.append(file_path)

    return media_files

def extract_audio_from_video(video_path, output_path):
    """Extract audio from video file using ffmpeg"""
    import subprocess

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'libmp3lame',  # MP3 codec
        '-ab', '128k',  # Bitrate
        '-ar', '44100',  # Sample rate
        '-y',  # Overwrite output
        str(output_path)
    ]

    print(f"Extracting audio from {video_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Audio extracted successfully: {output_path}")
        return True
    else:
        print(f"Failed to extract audio: {result.stderr}")
        return False

def main():
    print("Media Processing Tool")
    print("=" * 40)

    # Find media files
    media_files = find_video_files()

    if not media_files:
        print("No media files found in the project directory.")
        print("Please download a YouTube video/audio and place it in this folder.")
        print("Supported formats: MP4, M4V, MOV, AVI, MKV, WEBM, M4A, MP3, WAV")
        return

    print(f"Found {len(media_files)} media file(s):")
    for i, media in enumerate(media_files, 1):
        print(f"{i}. {media}")

    # Process each media file
    pipeline = AudioTranscriptionPipeline()

    for video_path in media_files:
        print(f"\nProcessing: {video_path}")

        # Create audio output path
        audio_path = video_path.parent / f"{video_path.stem}_audio.mp3"

        # Extract audio if needed
        if not audio_path.exists():
            if not extract_audio_from_video(video_path, audio_path):
                print(f"Skipping {video_path} due to audio extraction failure")
                continue

        # Transcribe with Whisper
        print(f"Transcribing {audio_path}...")
        transcript_data = pipeline.transcribe_audio(str(audio_path))

        if transcript_data:
            # Save transcript
            title = video_path.stem.replace('_', ' ')
            pipeline.save_transcript(transcript_data, f"{video_path.stem}_transcript")

            print(f"✅ Successfully processed: {video_path}")
            print(f"   Transcript saved as: {video_path.stem}_transcript.json")
            print(f"   Duration: {transcript_data['duration']:.2f} seconds")
            print(f"   Word count: {transcript_data['word_count']}")
        else:
            print(f"❌ Failed to transcribe: {video_path}")

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
