#!/usr/bin/env python3
"""
High-Performance Multi-Threaded Transcription for M2 MacBook Air
Utilizes all CPU cores, memory optimization, and parallel processing
"""

import os
import torch
import whisper
import threading
import concurrent.futures
import multiprocessing
from pathlib import Path
import logging
import time
import queue
from typing import List, Dict, Optional
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighPerformanceTranscriber:
    def __init__(self, model_size="medium", max_workers=None):
        self.model_size = model_size
        self.device = "cpu"
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())  # Use up to 8 cores

        # M2 CPU optimizations
        self._setup_m2_optimizations()

        # Threading components
        self.models = {}  # Thread-local models
        self.lock = threading.Lock()

        logger.info(f"Initialized high-performance transcriber with {self.max_workers} workers")

    def _setup_m2_optimizations(self):
        """M2 MacBook Air specific optimizations"""
        # CPU thread optimization
        cpu_count = multiprocessing.cpu_count()
        thread_count = min(8, cpu_count)  # M2 has 8 performance cores

        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)

        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(thread_count // 2)

        # Memory optimizations
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Aggressive memory management

        logger.info(f"M2 optimizations: {thread_count} CPU threads, memory optimization enabled")

    def _load_model_for_thread(self, thread_id: int):
        """Load a separate model instance for each thread"""
        if thread_id not in self.models:
            with self.lock:
                logger.info(f"Loading model for thread {thread_id}...")
                model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=None,
                    in_memory=False  # Memory efficient
                )
                self.models[thread_id] = model
                logger.info(f"Model loaded for thread {thread_id}")

        return self.models[thread_id]

    def transcribe_chunk_parallel(self, args) -> Dict:
        """Transcribe a single chunk with its own model instance"""
        audio_path, start_time, duration, chunk_id, thread_id = args

        try:
            model = self._load_model_for_thread(thread_id)

            logger.info(".1f")

            # Optimized transcription settings
            result = model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                fp16=False,  # FP32 for M2 CPU
                clip_timestamps=[start_time, start_time + duration]
            )

            return {
                "chunk_id": chunk_id,
                "start_time": start_time,
                "text": result["text"],
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in chunk {chunk_id}: {e}")
            return {
                "chunk_id": chunk_id,
                "start_time": start_time,
                "text": "",
                "success": False,
                "error": str(e)
            }

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration efficiently"""
        import subprocess
        import json

        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            data = json.loads(result.stdout)

            # Get duration from format or streams
            duration = float(data.get('format', {}).get('duration', 0))
            if duration == 0 and 'streams' in data:
                for stream in data['streams']:
                    if stream.get('codec_type') == 'audio':
                        duration = float(stream.get('duration', 0))
                        break

            return duration

        except Exception as e:
            logger.error(f"Could not get audio duration: {e}")
            return 0

    def create_transcription_tasks(self, audio_path: str, chunk_duration: int = 180) -> List:
        """Create parallel transcription tasks"""
        duration = self.get_audio_duration(audio_path)
        if duration == 0:
            return []

        tasks = []
        chunk_id = 0

        for start_time in range(0, int(duration), chunk_duration):
            actual_duration = min(chunk_duration, duration - start_time)
            thread_id = chunk_id % self.max_workers  # Distribute across workers

            tasks.append((
                audio_path,
                float(start_time),
                actual_duration,
                chunk_id,
                thread_id
            ))
            chunk_id += 1

        logger.info(f"Created {len(tasks)} transcription tasks across {self.max_workers} workers")
        return tasks

    def transcribe_audio_parallel(self, audio_path: str) -> Optional[str]:
        """Transcribe entire audio file using parallel processing"""
        start_time = time.time()

        # Create transcription tasks
        tasks = self.create_transcription_tasks(audio_path)
        if not tasks:
            return None

        logger.info(f"🎯 Starting parallel transcription with {self.max_workers} workers")

        # Execute tasks in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.transcribe_chunk_parallel, task): task for task in tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)

                # Progress update
                completed = len([r for r in results if r["success"]])
                total = len(tasks)
                logger.info(".1f")

        # Sort results by chunk_id and combine
        results.sort(key=lambda x: x["chunk_id"])
        full_transcript = " ".join([r["text"] for r in results if r["success"]])

        # Memory cleanup
        gc.collect()

        processing_time = time.time() - start_time
        logger.info(".1f")

        return full_transcript.strip()

def get_system_info():
    """Get system performance information"""
    cpu_count = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()

    print("🚀 System Performance Info:")
    print(f"   CPU Cores: {cpu_count}")
    print(".1f")
    print(".1f")
    print(f"   CPU Frequency: {cpu_freq.current:.0f} MHz" if cpu_freq else "   CPU Frequency: Unknown")
    print(f"   Using {min(8, cpu_count)} workers for parallel processing")
    print()

def main():
    print("🚀 High-Performance Multi-Threaded Transcription")
    print("=" * 55)
    get_system_info()

    # Find audio files
    audio_dir = Path("audio")
    if not audio_dir.exists():
        print("❌ No audio directory found!")
        return

    audio_files = list(audio_dir.glob("*.m4a")) + list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("❌ No audio files found!")
        return

    print(f"📁 Found {len(audio_files)} audio file(s)")
    print("🎯 Using parallel processing for maximum M2 performance")
    print()

    # Initialize high-performance transcriber
    transcriber = HighPerformanceTranscriber(
        model_size="medium",  # Good balance of speed vs accuracy
        max_workers=min(8, multiprocessing.cpu_count())
    )

    for audio_file in audio_files:
        print(f"🎵 Processing: {audio_file.name}")
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        print(".1f")

        start_time = time.time()

        # Parallel transcription
        transcript = transcriber.transcribe_audio_parallel(str(audio_file))

        if transcript:
            # Save results
            transcripts_dir = Path("transcripts")
            transcripts_dir.mkdir(exist_ok=True)

            # Save text transcript
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
                "parallel_workers": transcriber.max_workers,
                "optimized_for": "M2 MacBook Air",
                "processing_time_seconds": time.time() - start_time,
                "performance_mode": "multi-threaded_parallel"
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            processing_time = time.time() - start_time
            words_per_minute = len(transcript.split()) / (processing_time / 60)

            print("✅ Transcription completed!")
            print(".1f")
            print(".1f")
            print(f"📝 Words: {len(transcript.split()):,}")
            print(f"💾 Saved: {transcript_file}")
            print()
        else:
            print("❌ Transcription failed!")
            print()

if __name__ == "__main__":
    main()
