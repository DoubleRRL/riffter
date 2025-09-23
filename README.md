# Riffter

Custom LoRA-trained comedy bot that riffs like Cum Town. Built from scratch on an M2 Air, trained on 26k chunks of raw podcast dialogue.

## Overview

We took Microsoft's DialoGPT-small and LoRA fine-tuned it on Cum Town transcripts for ~19,600 steps. The result? A model that captures Nick Mullen's wild, unfiltered comedy style - deep cuts, absurd connections, and that raw "feels wrong but right" logic. No generic AI bullshit here.

The system serves up structured jokes and quick riffs through a clean web interface. Training completed successfully with LoRA adapters, running on Apple MPS for max M2 Air performance.

## Tech Stack

-   **Backend**: FastAPI, Python 3.13
-   **AI**: DialoGPT-small + LoRA fine-tuning (Hugging Face Transformers)
-   **Training**: ~19,600 steps on M2 Air with Apple MPS acceleration
-   **Frontend**: React + Vite
-   **Hardware**: Built and trained on M2 Air (8GB RAM)

## Getting Started

**What you need:**
- Python 3.8+ (we used 3.13)
- Node.js for the frontend

**Quick start:**
```bash
./start.sh
```

This fires up everything - backend on :8000, frontend on :5173.

**Manual setup:**

1. **Python environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Start services**:
   ```bash
   # Backend
   python src/api/main.py

   # Frontend
   cd frontend && npm run dev
   ```

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

**Start the app:**
```bash
./start.sh
```

The web interface will be at `http://localhost:5173`. Type in prompts and get Cum Town-style riffs back. The backend loads the LoRA-trained DialoGPT model automatically.

## Project Structure

```
riffter/
├── src/
│   ├── api/              # FastAPI backend
│   ├── generation/       # Model inference code
│   ├── training/         # LoRA training scripts
│   └── utils/            # Helper utilities
├── models/
│   └── cumtown_model/    # LoRA-trained DialoGPT + checkpoints
├── transcripts/          # Raw Cum Town transcripts
├── data/                 # Processed training data (26k chunks)
├── frontend/             # React web interface
├── docs/                 # Documentation
├── start.sh             # Quick startup script
└── start_training.sh    # Training launcher
```
