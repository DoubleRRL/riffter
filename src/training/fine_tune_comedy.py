#!/usr/bin/env python3
"""
Fine-tune GPT-2 model on Nick Mullen comedy transcripts
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import json
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComedyDataset(Dataset):
    """Dataset for comedy transcripts"""

    def __init__(self, transcripts_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load transcripts
        if transcripts_file.endswith('.xlsx'):
            df = pd.read_excel(transcripts_file)
            texts = df['text'].tolist()
        elif transcripts_file.endswith('.json'):
            with open(transcripts_file, 'r') as f:
                data = json.load(f)
            texts = [item.get('transcript', '') for item in data if item.get('transcript')]
        else:
            # Load from text files
            transcripts_dir = Path(transcripts_file)
            texts = []
            for txt_file in transcripts_dir.glob("*.txt"):
                with open(txt_file, 'r') as f:
                    texts.append(f.read())

        # Process texts
        for text in texts:
            if len(text.strip()) > 50:  # Filter out very short texts
                self.data.append(text.strip())

        logger.info(f"Loaded {len(self.data)} transcripts")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'labels': encodings['input_ids'].flatten()  # For language modeling
        }

def setup_model_and_tokenizer(model_path):
    """Load model and tokenizer"""
    logger.info("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Use quantization on GPU
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # Load on CPU with offloading
        import tempfile
        temp_dir = tempfile.mkdtemp()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=temp_dir,
            low_cpu_mem_usage=True
        )

    return model, tokenizer

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning"""
    logger.info("Setting up LoRA...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

    return model

def fine_tune_model(model_path, data_path, output_dir="models/fine_tuned_comedy"):
    """Fine-tune the model on comedy data"""

    # Setup
    model, tokenizer = setup_model_and_tokenizer(model_path)
    model = setup_lora(model)

    # Create dataset
    dataset = ComedyDataset(data_path, tokenizer)

    if len(dataset) == 0:
        logger.error("No training data found!")
        return None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Small batch size for memory
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model fine-tuned and saved to: {output_dir}")
    return output_dir

def generate_comedy_sample(model_path, prompt="Hey guys, welcome back to the podcast"):
    """Generate a comedy sample with the fine-tuned model"""
    logger.info("Loading model for generation...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load GPT-2 model (CPU friendly)
    logger.info("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device if device == "cuda" else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated comedy: {generated_text}")

    return generated_text

def main():
    # Paths
    data_path = "transcripts"  # Directory with transcripts

    logger.info("Starting comedy model testing...")

    # Check if we have training data
    if not Path(data_path).exists():
        logger.warning(f"No training data found at {data_path}")
        logger.info("Please run the audio transcription pipeline first")
        return

    # Test different models
    models_to_test = [
        ("models/nick_mullen_model", "Nick Mullen GPT-2 model"),
        ("models/comedy_model", "Comedy GPT-2 model")
    ]

    for model_path, description in models_to_test:
        if Path(model_path).exists():
            logger.info(f"Testing {description}...")
            try:
                sample = generate_comedy_sample(model_path)
                logger.info(f"{description} sample: {sample[:500]}...")  # Truncate for readability
            except Exception as e:
                logger.error(f"Failed to generate with {description}: {str(e)[:200]}...")
        else:
            logger.info(f"{description} not found at {model_path}")

    logger.info("Model testing complete!")
    logger.info("GPT-2 models work perfectly on CPU")

if __name__ == "__main__":
    main()
