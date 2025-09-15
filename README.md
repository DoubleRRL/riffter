# Riffter

AI-powered comedy generator that transcribes podcasts and creates Nick Mullen-style content.

## Features

- Downloads YouTube videos and extracts audio transcripts using Whisper
- Fine-tunes GPT-2 models on comedy transcripts
- Generates rambling, absurd comedy in Nick Mullen's style
- Web interface for testing and generating content
- CPU-only operation (no GPU required)

## Usage

```bash
# Generate comedy with existing models
python src/generation/quick_generate.py

# Process new videos for training data
python src/transcription/process_manual_video.py

# Train new models
python src/training/fine_tune_comedy.py

# Start web interface
python main.py                    # Backend on :8000
cd frontend && npm run dev       # Frontend on :3000
```

## Project Structure

```
riffter/
├── src/
│   ├── transcription/     # Video download & transcription
│   ├── training/         # Model fine-tuning scripts
│   ├── generation/       # Content generation
│   ├── api/              # FastAPI backend
│   └── utils/            # Helper scripts
├── tests/                # Test files
├── models/               # Trained AI models
├── transcripts/          # Generated transcripts
├── frontend/             # React web app
└── main.py              # Server entry point
```

## Status

Currently trained on Cum Town episodes. Generates authentic Nick Mullen-style comedy rants about random topics.

## Dependencies

- Python 3.8+
- Whisper (OpenAI)
- Transformers (Hugging Face)
- yt-dlp
- FastAPI
- React
