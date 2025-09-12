#!/usr/bin/env python3
"""
Simple script to generate Nick Mullen-style comedy using fine-tuned GPT-2 models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_comedy_content(model_path, prompt="Hey guys, welcome back to the podcast", num_samples=3, max_length=150):
    """Generate comedy content using the specified model"""

    logger.info(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Generating {num_samples} comedy samples...")

    for i in range(num_samples):
        logger.info(f"\n--- Sample {i+1} ---")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1
            )

        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text[len(prompt):].strip())  # Remove the prompt from output

def main():
    # Available models
    models = {
        "1": ("models/nick_mullen_model", "Nick Mullen GPT-2 Model"),
        "2": ("models/comedy_model", "General Comedy GPT-2 Model")
    }

    print("🎭 Nick Mullen Comedy Generator 🎭")
    print("==================================")
    print("Available models:")
    for key, (path, desc) in models.items():
        print(f"{key}. {desc}")

    choice = input("\nSelect model (1 or 2): ").strip()

    if choice not in models:
        print("Invalid choice!")
        return

    model_path, description = models[choice]
    print(f"\nUsing: {description}")

    # Get custom prompt
    prompt = input("Enter a prompt (or press Enter for default): ").strip()
    if not prompt:
        prompt = "Hey guys, welcome back to the podcast"

    # Get number of samples
    try:
        num_samples = int(input("How many samples to generate? (default 3): ").strip() or "3")
    except ValueError:
        num_samples = 3

    # Generate content
    generate_comedy_content(model_path, prompt, num_samples)

    print("\n✨ Generation complete! ✨")

if __name__ == "__main__":
    main()
