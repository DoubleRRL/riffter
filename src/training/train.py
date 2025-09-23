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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    model_name: str = "microsoft/DialoGPT-small",  # Smaller model (117M params) that fits in memory
    output_dir: str = "models/cumtown_model",
    epochs: int = 3,
    batch_size: int = 2,  # Increased batch size for chunked data
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 2,  # Reduced for larger batch size
    resume_from_checkpoint: str = None
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

    # Configure LoRA for memory-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # Rank dimension - higher = more capacity, lower = more efficient
        lora_alpha=32,  # Scaling factor
        target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 attention modules
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Prepare model for k-bit training (8-bit quantization)
    if device_type == "cuda":
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show parameter reduction

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

    # Training arguments - optimized for LoRA training
    training_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size * 2,  # Can use larger batch size with LoRA
        "gradient_accumulation_steps": gradient_accumulation_steps // 2,  # Reduce accumulation since batch size increased
        "learning_rate": learning_rate * 2,  # LoRA can handle higher learning rates
        "logging_dir": './logs',
        "logging_steps": 10,
        "save_steps": 50,  # More frequent saves for LoRA (smaller checkpoints)
        "save_total_limit": 5,  # Keep more checkpoints
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        "remove_unused_columns": False,
        "warmup_steps": 50,  # Shorter warmup for LoRA
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
    logger.info("Starting LoRA fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the LoRA adapters and tokenizer
    trainer.save_model(output_dir)
    logger.info(f"LoRA adapters saved to {output_dir}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Also save the base model config for inference
    model.config.save_pretrained(output_dir)

def main():
    training_data_path = "data/cumtown_chunked_training_data.json"

    if not Path(training_data_path).exists():
        logger.error(f"Training data not found: {training_data_path}")
        logger.error("Run chunking script first")
        return

    # Resume from latest checkpoint if it exists
    model_dir = Path("models/cumtown_model")
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if checkpoints:
        # Find the checkpoint with the highest step number
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        print(f"🔄 Resuming from checkpoint: {latest_checkpoint.name}")
        resume_from_checkpoint = str(latest_checkpoint)
    else:
        print("🆕 Starting fresh training")
        resume_from_checkpoint = None

    train_model(
        data_path=training_data_path,
        epochs=3,
        batch_size=2,
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
        resume_from_checkpoint=resume_from_checkpoint
    )

if __name__ == "__main__":
    main()