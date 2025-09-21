# Riffter

AI-powered comedy sidekick for generating edgy, Nick Mullen-style riffs and jokes.

## Overview

This project uses Ollama to power a local AI comedian. You give it a topic, and it generates structured jokes and quick riffs based on a specific, edgy comedic style. It's all managed through a simple web interface.

## Tech Stack

-   **Backend**: FastAPI, Python
-   **AI**: Ollama with `Godmoded/llama3-lexi-uncensored`
-   **Frontend**: React, Vite

## Getting Started

**What you need:**
- Python 3.8+ (you're probably already running this)
- Node.js (download from nodejs.org if you don't have it)
- Ollama installed (grab it from ollama.ai) [[memory:4807211]]

**Quick start (use the script):**
```bash
./start.sh
```

This script handles everything - sets up the virtual environment, installs dependencies, and starts both the backend and frontend. Boom, you're done.

**Manual setup (if you hate easy things):**

1. **Install Ollama model** (optional, but recommended):
   ```bash
   ollama pull Godmoded/llama3-lexi-uncensored
   ```

2. **Set up Python stuff**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Start the services**:
   ```bash
   # Terminal 1 - Backend
   python src/api/main.py

   # Terminal 2 - Frontend
   cd frontend
   npm run dev
   ```

The backend runs on `http://localhost:8000`, frontend on `http://localhost:5173`.

## The Comedian's Personality

To get the best results, you're not talking to a generic LLM. You're talking to a very specific type of comedian. Keep these guidelines in mind:

-   **Concise**: Max 15 words per line.
-   **Metaphors over Similes**: Punchy and unexpected, not wordy.
-   **Style**: Think Nick Mullen—deep cuts, wild imagery, absurd connections.
-   **Logic**: Aims for "wrong but feels right" conclusions.
-   **Language**: Uses modern slang if it serves the joke.
-   **Sound**: Embraces raw, phonetic sounds ("buh," "cuh," "fuh") for comedic effect. It's funnier to say "you're just gay drinking cum" than "this gay guy loves semen."

This persona is baked into the prompts to ensure the output doesn't sound like a sanitized, corporate HR bot.

## Usage

**For development:**
```bash
# Quick start (recommended)
./start.sh

# Or manual start
python src/api/main.py          # Backend on :8000
cd frontend && npm run dev      # Frontend on :5173
```

**For content creation:**
```bash
# Process videos and extract transcripts
python src/transcription/process_manual_video.py

# Generate comedy content
python src/generation/inference.py
```

## Project Structure

```
riffter/
├── src/
│   ├── transcription/     # Video download & audio processing
│   ├── generation/        # AI comedy generation
│   ├── api/              # FastAPI backend
│   └── utils/            # Helper utilities
├── models/               # AI models (when trained)
├── transcripts/          # Generated transcripts
├── data/                 # Training datasets
├── frontend/             # React web interface
├── docs/                 # Documentation
└── start.sh             # Quick startup script
```
