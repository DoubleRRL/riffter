#!/usr/bin/env python3
"""
Fine-tuning script for the Nick Mullen comedy model using Hugging Face Transformers
Uses Llama-3.1-8B-Instruct as the base model
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_path: str) -> Dataset:
    """Load the Cum Town training data and prepare it for training"""
    try:
        # Load the processed data
        with open(data_path, 'r') as f:
            import json
            data = json.load(f)

        # Extract text content
        texts = [item["text"] for item in data if "text" in item and len(item["text"].strip()) > 50]

        logger.info(f"Loaded {len(texts)} training samples")

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        return dataset
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def tokenize_function(examples, tokenizer):
    """Tokenize the text for training"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

def train_model(
    data_path: str,
    model_name: str = "microsoft/DialoGPT-medium",  # Already cached and works well for conversations
    output_dir: str = "models/nick_mullen_model",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1
):
    """Fine-tune the language model on Cum Town data"""

    # Get Hugging Face token (only needed for some models)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    # Load dataset
    dataset = load_and_prepare_data(data_path)

    # Load tokenizer
    tokenizer_kwargs = {}
    if hf_token and "meta-llama" in model_name:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for compute availability
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device_type = "cuda" if cuda_available else ("mps" if mps_available else "cpu")

    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"MPS available: {mps_available}")
    logger.info(f"Using device: {device_type}")

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,  # Use float32 for all devices for compatibility
    }

    if hf_token and "meta-llama" in model_name:
        model_kwargs["token"] = hf_token

    if device_type == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["load_in_8bit"] = True  # 8-bit quantization for memory efficiency on GPU
        logger.info("Using 8-bit quantization for CUDA GPU")
    else:
        # For MPS and CPU, use disk offloading to handle large models
        model_kwargs["device_map"] = "auto"  # This enables disk offloading
        model_kwargs["torch_dtype"] = torch.float32
        logger.info("Using disk offloading for memory efficiency")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Training arguments - different for CPU vs GPU
    training_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "logging_dir": './logs',
        "logging_steps": 10,
        "save_steps": 100,  # Less frequent saves for smaller model
        "save_total_limit": 3,  # Keep more checkpoints
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        "remove_unused_columns": False,
        "warmup_steps": 100,  # Warmup for stable training
        "weight_decay": 0.01,  # Small weight decay
    }

    if device_type == "cuda":
        training_args_kwargs.update({
            "bf16": True,  # Use bfloat16 for CUDA training
            "fp16": False,  # Don't use fp16 with 8-bit quantization
        })
    elif device_type == "mps":
        training_args_kwargs.update({
            "bf16": False,  # MPS doesn't support bfloat16 well
            "fp16": False,  # MPS has limited fp16 support
        })
        logger.info("MPS training detected - using float32")
    else:
        training_args_kwargs.update({
            "bf16": False,  # No bfloat16 on CPU
            "fp16": False,  # No fp16 on CPU
        })
        logger.info("CPU training detected - this will be slow and may fail")

    training_args = TrainingArguments(**training_args_kwargs)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Start training
    logger.info("Starting model training...")
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

def main():
    training_data_path = "data/cumtown_training_data.json"

    if not Path(training_data_path).exists():
        logger.error(f"Training data not found: {training_data_path}")
        logger.error("Run: python src/training/process_cumtown_data.py")
        return

    train_model(
        data_path=training_data_path,
        epochs=3,
        batch_size=4,  # Can use larger batch size for smaller DialoGPT-medium model
        learning_rate=2e-5,
        gradient_accumulation_steps=4  # Effective batch size = 4
    )

if __name__ == "__main__":
    main()