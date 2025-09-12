# Riffter

A pipeline that transcribes comedy podcasts, fine-tunes AI models on Nick Mullen's style, and generates new comedy content. Currently processing Cum Town episodes to create Nick Mullen-style AI comedy.

## What It Does

- **Transcribes Audio**: Downloads YouTube videos and uses Whisper to create transcripts
- **Fine-tunes Models**: Trains GPT-2 models on Nick Mullen comedy transcripts
- **Generates Comedy**: Creates new content in Nick Mullen's rambling, absurd style
- **Web Interface**: Simple React frontend for testing generation

## Quick Start

```bash
# Generate comedy with existing models
python quick_generate.py

# Process new audio/video files
python process_manual_video.py

# Train models on transcripts
python fine_tune_comedy.py

# Run web interface
python main.py  # Backend on :8000
cd frontend && npm run dev  # Frontend on :3000
```

## Files You Need

- `audio/` - Raw audio files from YouTube downloads
- `transcripts/` - Whisper-generated transcripts from Cum Town episodes
- `models/` - Fine-tuned GPT-2 models (nick_mullen_model, comedy_model)
- `venv/` - Python environment with all dependencies

## Current Status

Working on a 2-hour Cum Town compilation. Models generate authentic Nick Mullen-style rambling comedy about random topics. CPU-friendly, no GPU required.

## Tech Stack

- Python/Whisper for transcription
- GPT-2 with LoRA fine-tuning
- FastAPI backend
- React frontend
- yt-dlp for YouTube downloads
