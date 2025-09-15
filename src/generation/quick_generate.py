#!/usr/bin/env python3
"""
Quick demo script to generate Nick Mullen-style comedy content
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_samples(model_path="/Users/RRL_1/.llama/checkpoints/Llama3.1-8B", prompt="Hey guys, welcome back to the podcast", num_samples=3):
    """Generate comedy samples"""

    logger.info(f"Loading Llama 3.1-8B comedy model...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🎭 Llama 3.1-8B Comedy Generator 🎭")
    print("=" * 40)

    for i in range(num_samples):
        print(f"\n📣 Sample {i+1}:")
        print("-" * 20)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1
            )

        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        content = generated_text[len(prompt):].strip()

        # Clean up and format
        if content:
            print(content)
        else:
            print("(Generation was too short)")

    print("\n✨ Keep riffing! ✨")

if __name__ == "__main__":
    generate_samples()
