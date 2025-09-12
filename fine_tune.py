#!/usr/bin/env python3
"""
Fine-tune a small language model on Cum Town and Adam Friedland Show Podcast style
Uses PEFT for efficient training on limited hardware
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComedyFineTuner:
    def __init__(self):
        self.data_dir = Path("data")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        # Use a small, efficient model for M2 Air
        self.model_name = "gpt2"  # Small and efficient for fine-tuning
        self.output_dir = self.model_dir / "comedy_model"

    def load_training_data(self):
        """Load full transcripts from scraped data"""
        training_data_file = self.data_dir / "full_transcripts_training_data.json"

        if not training_data_file.exists():
            logger.error(f"Training data not found: {training_data_file}")
            logger.info("Run scrape_transcripts.py first to generate training data")
            return None

        with open(training_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} training samples from {len(set(item['podcast'] for item in data))} podcasts")

        # Convert to format expected by transformers
        texts = [item['text'] for item in data]

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        return dataset

    def prepare_model_and_tokenizer(self):
        """Load model and tokenizer with memory optimizations"""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model optimized for M2 (Apple Silicon)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for M2 compatibility
            device_map="cpu",  # Use CPU for now (M2 can handle it)
        )

        logger.info("Model loaded successfully")
        return model, tokenizer

    def setup_peft(self, model):
        """Set up Parameter-Efficient Fine-Tuning"""
        logger.info("Setting up PEFT...")

        # LoRA configuration for efficient fine-tuning
        lora_config = LoraConfig(
            r=8,  # Smaller rank for memory efficiency
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"]  # GPT-2 attention modules
        )

        # Apply PEFT to model
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    def tokenize_function(self, tokenizer):
        """Create tokenization function"""
        def tokenize(examples):
            # Tokenize the texts
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            return tokenized
        return tokenize

    def fine_tune(self):
        """Main fine-tuning workflow"""
        logger.info("Starting fine-tuning process...")

        # Load training data
        dataset = self.load_training_data()
        if dataset is None:
            return False

        # Prepare model and tokenizer
        model, tokenizer = self.prepare_model_and_tokenizer()

        # Setup PEFT
        model = self.setup_peft(model)

        # Split dataset
        train_dataset = dataset.train_test_split(test_size=0.1)["train"]
        eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

        # Tokenize datasets
        tokenize_fn = self.tokenize_function(tokenizer)
        tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Training arguments - optimized for M2 Air
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=2,  # Fewer epochs for quick testing
            per_device_train_batch_size=1,  # Very small batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,  # Effective batch size = 2
            learning_rate=5e-4,  # Higher learning rate for faster convergence
            weight_decay=0.01,
            logging_steps=5,
            save_steps=25,
            eval_steps=25,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            # Remove fp16 for M2 compatibility
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)

        logger.info(f"Model saved to: {self.output_dir}")
        return True

    def generate_sample(self):
        """Generate a sample riff with the fine-tuned model"""
        if not self.output_dir.exists():
            logger.error("No fine-tuned model found. Run fine_tune() first.")
            return None

        logger.info("Loading fine-tuned model for generation...")

        try:
            from transformers import pipeline

            # Load the fine-tuned model
            generator = pipeline(
                "text-generation",
                model=str(self.output_dir),
                tokenizer=str(self.output_dir),
                device=0 if torch.cuda.is_available() else -1
            )

            # Generate a sample
            prompt = "You are Nick Mullen from Cum Town. Generate an absurd riff about dating apps:"
            result = generator(
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )

            generated_text = result[0]["generated_text"]
            # Extract just the generated part after the prompt
            riff = generated_text[len(prompt):].strip()

            logger.info(f"Generated riff: {riff}")
            return riff

        except Exception as e:
            logger.error(f"Error generating sample: {e}")
            return None

if __name__ == "__main__":
    tuner = ComedyFineTuner()

    # Ask user what they want to do
    print("Choose an option:")
    print("1. Fine-tune model")
    print("2. Generate sample with existing model")
    print("3. Both")

    choice = input("Enter choice (1-3): ").strip()

    if choice in ["1", "3"]:
        success = tuner.fine_tune()
        if not success:
            print("Fine-tuning failed. Check the logs above.")

    if choice in ["2", "3"]:
        sample = tuner.generate_sample()
        if sample:
            print(f"\nSample riff: {sample}")
        else:
            print("Could not generate sample.")
