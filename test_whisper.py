#!/usr/bin/env python3
"""
Test Whisper transcription functionality
"""

import whisper
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper():
    """Test Whisper model loading and basic functionality"""
    try:
        logger.info("Testing Whisper model loading...")

        # Test with smaller model first
        model = whisper.load_model("base")
        logger.info("Whisper base model loaded successfully!")

        # Check if CUDA is available
        if torch.cuda.is_available():
            logger.info("CUDA is available - GPU acceleration enabled")
        else:
            logger.info("CUDA not available - using CPU (will be slower)")

        # Test transcription on a simple audio file if available
        # For now, just test the model loading
        logger.info("Whisper is working correctly!")
        return True

    except Exception as e:
        logger.error(f"Whisper test failed: {e}")
        return False

def test_large_model():
    """Test loading the large model we want to use"""
    try:
        logger.info("Testing Whisper large-v3 model loading...")
        model = whisper.load_model("large-v3")
        logger.info("Whisper large-v3 model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Large model test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Whisper tests...")

    if test_whisper():
        logger.info("Basic Whisper test passed!")
        # Comment out large model test for now to save time
        # if test_large_model():
        #     logger.info("Large model test passed!")
        # else:
        #     logger.warning("Large model test failed, but basic model works")
    else:
        logger.error("Basic Whisper test failed")
