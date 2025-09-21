#!/usr/bin/env python3
"""
Test script to verify Hugging Face authentication and model download
"""

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hf_access():
    """Test Hugging Face access and model download"""

    # Check token
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("No Hugging Face token found in environment variables")
        logger.error("Make sure HUGGINGFACE_TOKEN or HF_TOKEN is set in .env file")
        return False

    logger.info("Found Hugging Face token")

    try:
        # Test downloading tokenizer (smaller, faster)
        logger.info("Testing tokenizer download...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small"
        )
        logger.info("✅ Tokenizer downloaded successfully")

        # Test downloading a small part of the model
        logger.info("Testing model download (this may take a moment)...")

        # Check for compute availability
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        device_type = "cuda" if cuda_available else ("mps" if mps_available else "cpu")

        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"MPS available: {mps_available}")
        logger.info(f"Using device: {device_type}")

        model_kwargs = {
            "token": hf_token,
            "torch_dtype": torch.float32,  # Use float32 for compatibility
            "low_cpu_mem_usage": True
        }

        if device_type == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["load_in_8bit"] = True
            logger.info("Using 8-bit quantization for CUDA GPU")
        elif device_type == "mps":
            # MPS still runs out of memory, use CPU with disk offloading
            model_kwargs["device_map"] = "auto"  # This will offload to disk
            model_kwargs["torch_dtype"] = torch.float32
            logger.info("Using disk offloading (MPS not enough memory)")
        else:
            # For CPU, use disk offloading to save RAM
            model_kwargs["device_map"] = "auto"  # This will offload to disk
            model_kwargs["torch_dtype"] = torch.float32
            logger.info("Using disk offloading for CPU")

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            **model_kwargs
        )
        logger.info("✅ Model downloaded successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Error accessing Hugging Face: {e}")
        if "401" in str(e):
            logger.error("This is likely an authentication error. Check your token.")
        elif "403" in str(e):
            logger.error("Access denied. You may not have permission to access this model.")
        return False

if __name__ == "__main__":
    print("Testing Hugging Face access for DialoGPT-small...")
    success = test_hf_access()

    if success:
        print("\n🎉 Success! You can now run the training script.")
        print("Run: python src/training/train.py")
    else:
        print("\n❌ Failed. Check your token and permissions.")
