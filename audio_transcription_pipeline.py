#!/usr/bin/env python3
"""
Audio Transcription Pipeline for Nick Mullen Comedy Content
Downloads YouTube videos, transcribes with Whisper, and prepares for Mistral fine-tuning
"""

import os
import json
import pandas as pd
import subprocess
import whisper
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging
import gradio as gr
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriptionPipeline:
    def __init__(self, audio_dir: str = "audio", transcripts_dir: str = "transcripts"):
        self.audio_dir = Path(audio_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.audio_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)

        # Initialize Whisper model (we'll load it when needed)
        self.whisper_model = None
        self.model_size = "large-v3"  # Best quality for comedy transcription

    def load_whisper_model(self):
        """Load Whisper model if not already loaded"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        return self.whisper_model

    def download_youtube_audio(self, url: str, output_filename: str) -> Optional[str]:
        """Download YouTube video as audio using yt-dlp"""
        try:
            output_path = self.audio_dir / f"{output_filename}.m4a"

            # yt-dlp command for best audio quality (use full path in venv)
            yt_dlp_path = os.path.join(os.path.dirname(__file__), "venv", "bin", "yt-dlp")
            cmd = [
                yt_dlp_path,
                "--extract-audio",
                "--audio-format", "m4a",
                "--audio-quality", "0",  # Best quality
                "--output", str(output_path),
                "--no-playlist",
                "--age-limit", "99",  # Allow age-restricted content
                "--embed-subs",  # Try to get subtitles if available
                url
            ]

            logger.info(f"Downloading audio from: {url}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.exists():
                logger.info(f"Successfully downloaded: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to download {url}: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[Dict]:
        """Transcribe audio file using Whisper"""
        try:
            model = self.load_whisper_model()

            logger.info(f"Transcribing: {audio_path}")

            # Transcribe with Whisper
            result = model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False,
                temperature=0.0,  # More deterministic
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True
            )

            # Clean up the transcript text
            transcript_text = self.clean_transcript_text(result["text"])

            transcript_data = {
                "audio_file": audio_path,
                "transcript": transcript_text,
                "duration": result.get("duration", 0),
                "language": result.get("language", "en"),
                "segments": result.get("segments", []),
                "word_count": len(transcript_text.split()),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Transcription complete: {len(transcript_text)} characters")
            return transcript_data

        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return None

    def clean_transcript_text(self, text: str) -> str:
        """Clean and normalize transcript text for comedy content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove timestamps and artifacts (Whisper sometimes adds these)
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\[\d{2}:\d{2}\]', '', text)

        # Remove common filler words that might be artifacts
        text = re.sub(r'\s+', ' ', text)

        # Clean up speaker transitions for comedy (Nick Mullen specific)
        text = re.sub(r'(Nick|Adam|Joe|Brian|Host)[\s:]+', r'\1: ', text, flags=re.IGNORECASE)

        return text.strip()

    def save_transcript(self, transcript_data: Dict, filename: str):
        """Save transcript to JSON and text files"""
        base_filename = Path(filename).stem

        # Save as JSON
        json_path = self.transcripts_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        # Save as text file
        txt_path = self.transcripts_dir / f"{base_filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_data['transcript'])

        logger.info(f"Saved transcript: {json_path}")
        return str(json_path), str(txt_path)

    def process_video(self, url: str, title: str) -> Optional[Dict]:
        """Complete pipeline: download -> transcribe -> save"""
        try:
            # Create safe filename
            safe_title = re.sub(r'[^\w\-_\.]', '_', title)
            audio_path = self.download_youtube_audio(url, safe_title)

            if not audio_path:
                return None

            # Transcribe
            transcript_data = self.transcribe_audio(audio_path)

            if not transcript_data:
                return None

            # Add metadata
            transcript_data.update({
                "source_url": url,
                "title": title,
                "processed_at": datetime.now().isoformat()
            })

            # Save
            self.save_transcript(transcript_data, f"{safe_title}_transcript")

            return transcript_data

        except Exception as e:
            logger.error(f"Error processing video {url}: {e}")
            return None

    def create_training_dataset(self, transcripts: List[Dict]) -> pd.DataFrame:
        """Create a dataset suitable for Mistral fine-tuning"""
        training_data = []

        for transcript in transcripts:
            # Split long transcripts into smaller chunks for better training
            text = transcript['transcript']
            chunks = self.split_into_chunks(text, max_words=512)

            for i, chunk in enumerate(chunks):
                training_sample = {
                    "text": chunk,
                    "source": transcript.get("title", "Unknown"),
                    "url": transcript.get("source_url", ""),
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "word_count": len(chunk.split()),
                    "style": "comedy_improvisation"  # Label for fine-tuning
                }
                training_data.append(training_sample)

        df = pd.DataFrame(training_data)
        logger.info(f"Created training dataset with {len(training_data)} samples")
        return df

    def split_into_chunks(self, text: str, max_words: int = 512) -> List[str]:
        """Split text into chunks suitable for training"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())

        return chunks

    def save_dataset_for_training(self, df: pd.DataFrame, filename: str = "mistral_training_data.xlsx"):
        """Save dataset in format ready for Mistral fine-tuning"""
        output_path = self.transcripts_dir / filename
        df.to_excel(output_path, index=False)
        logger.info(f"Saved training dataset: {output_path}")
        return str(output_path)

    def create_review_interface(self, transcripts: List[Dict]):
        """Create a Gradio interface for reviewing transcripts before training"""
        def display_transcript(index):
            if 0 <= index < len(transcripts):
                transcript = transcripts[index]
                return (
                    transcript.get('title', 'Unknown'),
                    transcript.get('transcript', ''),
                    f"Word count: {transcript.get('word_count', 0)}",
                    f"Duration: {transcript.get('duration', 0):.2f} seconds"
                )
            return "Index out of range", "", "", ""

        def approve_transcript(index):
            if 0 <= index < len(transcripts):
                transcripts[index]['approved'] = True
                return f"Approved transcript {index}"
            return "Index out of range"

        with gr.Blocks(title="Transcript Review Interface") as interface:
            gr.Markdown("# Nick Mullen Comedy Transcript Review")
            gr.Markdown("Review and approve transcripts before fine-tuning Mistral")

            with gr.Row():
                index_slider = gr.Slider(0, len(transcripts)-1, step=1, label="Transcript Index")

            with gr.Row():
                title_display = gr.Textbox(label="Title", interactive=False)
                word_count_display = gr.Textbox(label="Word Count", interactive=False)
                duration_display = gr.Textbox(label="Duration", interactive=False)

            transcript_display = gr.Textbox(
                label="Transcript",
                lines=20,
                interactive=False
            )

            approve_btn = gr.Button("Approve This Transcript")

            # Event handlers
            index_slider.change(
                display_transcript,
                inputs=[index_slider],
                outputs=[title_display, transcript_display, word_count_display, duration_display]
            )

            approve_btn.click(
                approve_transcript,
                inputs=[index_slider],
                outputs=[gr.Textbox(label="Status")]
            )

            # Load first transcript
            interface.load(
                lambda: (transcripts[0].get('title', ''), transcripts[0].get('transcript', ''),
                        f"Word count: {transcripts[0].get('word_count', 0)}",
                        f"Duration: {transcripts[0].get('duration', 0):.2f} seconds"),
                outputs=[title_display, transcript_display, word_count_display, duration_display]
            )

        return interface


def main():
    # URLs to process
    urls = [
        ("https://www.youtube.com/watch?v=yzJZtaWFB_s", "Nick Mullen Impressions Compilation"),
        # Add more URLs from the playlist here when you get them
    ]

    pipeline = AudioTranscriptionPipeline()

    transcripts = []

    # Process each video
    for url, title in urls:
        logger.info(f"Processing: {title}")
        transcript_data = pipeline.process_video(url, title)

        if transcript_data:
            transcripts.append(transcript_data)

    if transcripts:
        # Create training dataset
        training_df = pipeline.create_training_dataset(transcripts)

        # Save for training
        dataset_path = pipeline.save_dataset_for_training(training_df)

        # Create review interface
        logger.info("Creating review interface...")
        review_interface = pipeline.create_review_interface(transcripts)

        # Launch review interface
        review_interface.launch(share=False, server_name="localhost", server_port=7860)

        logger.info(f"Processing complete! Dataset saved to: {dataset_path}")
        logger.info("Review interface launched at http://localhost:7860")
    else:
        logger.warning("No transcripts were successfully generated")


if __name__ == "__main__":
    main()
