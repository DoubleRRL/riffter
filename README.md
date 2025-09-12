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
python src/generation/quick_generate.py

# Process new audio/video files
python src/transcription/process_manual_video.py

# Train models on transcripts
python src/training/fine_tune_comedy.py

# Run web interface
python main.py  # Backend on :8000
cd frontend && npm run dev  # Frontend on :3000
```

## Project Structure

```
riffter/
├── src/                    # Source code
│   ├── transcription/      # Audio/video transcription scripts
│   ├── training/          # Model fine-tuning scripts
│   ├── generation/        # Content generation scripts
│   ├── api/               # FastAPI server
│   └── utils/             # Utility scripts
├── tests/                 # Test files
├── docs/                  # Documentation
├── audio/                 # Raw audio files
├── transcripts/           # Generated transcripts
├── models/                # Fine-tuned AI models
├── frontend/              # React web interface
└── main.py               # Entry point
```

## Current Status

Working on a 2-hour Cum Town compilation. Models generate authentic Nick Mullen-style rambling comedy about random topics. CPU-friendly, no GPU required.

## Tech Stack

- Python/Whisper for transcription
- GPT-2 with LoRA fine-tuning
- FastAPI backend
- React frontend
- yt-dlp for YouTube downloads
