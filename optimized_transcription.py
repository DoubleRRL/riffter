#!/usr/bin/env python3
"""
Optimized transcription for M2 MacBook Air - faster processing
"""

import os
import torch
import whisper
from pathlib import Path
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTranscriber:
    def __init__(self, model_size="medium"):  # Changed from large-v3 to medium
        self.model_size = model_size
        self.model = None
        self.device = "cpu"  # M2 CPU is excellent for this

        # M2 optimizations
        os.environ["OMP_NUM_THREADS"] = "8"  # Use 8 CPU cores
        os.environ["MKL_NUM_THREADS"] = "8"
        torch.set_num_threads(8)  # PyTorch threads

    def load_model(self):
        """Load optimized Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper {self.model_size} model (optimized for M2)...")

            # M2-specific optimizations
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=None,
                in_memory=False  # Save memory
            )

            logger.info("Model loaded with M2 optimizations!")
        return self.model

    def transcribe_chunk(self, audio_path: str, start_time: float = 0, duration: float = 300) -> Optional[str]:
        """Transcribe a 5-minute chunk for better performance"""
        try:
            model = self.load_model()

            logger.info(f"Transcribing chunk: {start_time/60:.1f}-{start_time/60 + 5:.1f} minutes")

            # Optimized transcription settings for M2
            result = model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False,
                temperature=0.0,  # More deterministic, faster
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                initial_prompt=None,  # Faster without prompt
                # M2-specific optimizations
                fp16=False,  # Use FP32 for M2 CPU
                # Process in chunks for memory efficiency
                clip_timestamps=[start_time, start_time + duration]
            )

            return result["text"]

        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return None

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        import subprocess
        import json

        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return float(data['format']['duration'])

        except Exception as e:
            logger.error(f"Could not get audio duration: {e}")
            return 0

    def transcribe_audio_optimized(self, audio_path: str) -> Optional[str]:
        """Transcribe entire audio file using optimized chunked approach"""
        duration = self.get_audio_duration(audio_path)
        logger.info(f"Audio duration: {duration/60:.1f} minutes")

        if duration == 0:
            return None

        full_transcript = ""
        chunk_duration = 300  # 5 minutes per chunk

        for start_time in range(0, int(duration), chunk_duration):
            chunk_text = self.transcribe_chunk(audio_path, start_time, min(chunk_duration, duration - start_time))

            if chunk_text:
                full_transcript += chunk_text + " "
                logger.info(f"Progress: {start_time/60:.1f}/{duration/60:.1f} minutes completed")
            else:
                logger.warning(f"Failed to transcribe chunk at {start_time}")

        return full_transcript.strip()

def main():
    print("🎯 Optimized Transcription for M2 MacBook Air")
    print("=" * 50)

    # Find audio files
    audio_dir = Path("audio")
    if not audio_dir.exists():
        print("❌ No audio directory found!")
        return

    audio_files = list(audio_dir.glob("*.m4a")) + list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("❌ No audio files found in audio/ directory!")
        print("   Place your downloaded audio files there first.")
        return

    print(f"📁 Found {len(audio_files)} audio file(s)")

    # Initialize optimized transcriber
    transcriber = OptimizedTranscriber(model_size="medium")  # Faster than large-v3

    for audio_file in audio_files:
        print(f"\n🎵 Processing: {audio_file.name}")
        print(f"   Size: {audio_file.stat().st_size / (1024*1024):.1f} MB")

        start_time = time.time()

        # Transcribe with optimizations
        transcript = transcriber.transcribe_audio_optimized(str(audio_file))

        if transcript:
            # Save transcript
            transcripts_dir = Path("transcripts")
            transcripts_dir.mkdir(exist_ok=True)

            transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript)

            # Save JSON metadata
            json_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
            import json
            metadata = {
                "audio_file": str(audio_file),
                "transcript": transcript,
                "word_count": len(transcript.split()),
                "model": transcriber.model_size,
                "optimized_for": "M2 MacBook Air",
                "processing_time_seconds": time.time() - start_time
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            processing_time = time.time() - start_time
            print(".1f")
            print(f"✅ Saved: {transcript_file}")
            print(f"📝 Words: {len(transcript.split()):,}")
        else:
            print("❌ Transcription failed!")

if __name__ == "__main__":
    main()
