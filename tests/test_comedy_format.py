#!/usr/bin/env python3
"""
Test comedy generation with proper joke format structure
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComedyFormatTester:
    def __init__(self, model_path="models/nick_mullen_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the GPT-2 model"""
        logger.info(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Model loaded successfully!")

    def generate_with_format(self, prompt="dating disasters", temperature=0.8):
        """Generate comedy with proper joke format structure"""

        # Create a structured prompt that encourages the joke format
        structured_prompt = f"""Hey guys, welcome back to the podcast. Today we're talking about {prompt}.

Let me tell you a joke:

Premise: """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = self.tokenizer(structured_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part
        content = generated_text[len(structured_prompt):].strip()

        return self.format_joke_output(content, prompt)

    def format_joke_output(self, content, original_prompt):
        """Format the generated content into proper joke structure"""

        print("🎭 NICK MULLEN COMEDY GENERATOR 🎭")
        print("=" * 60)
        print(f"Topic: {original_prompt}")
        print("-" * 60)

        # Split content into sentences/sections
        import re
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Try to structure into joke format
        print("📝 PREMISE:")
        if sentences:
            premise = sentences[0] + "."
            print(f"   {premise}")
        else:
            print("   (No premise generated)")

        print("\n💥 PUNCHLINE:")
        if len(sentences) > 1:
            punchline = sentences[1] + "."
            print(f"   {punchline}")
        else:
            print("   (No punchline generated)")

        # Additional punchlines
        if len(sentences) > 2:
            print("\n💥 PUNCHLINE:")
            for i, sentence in enumerate(sentences[2:5], 1):  # Next 3 sentences
                if i <= 3:
                    print(f"   {sentence}.")

        # Another angle
        if len(sentences) > 5:
            print("\n🔄 ANOTHER ANGLE:")
            print(f"   {sentences[5]}.")

        # Tags (punchlines without setup)
        print("\n🏷️  TAGS:")
        tag_candidates = sentences[6:] if len(sentences) > 6 else ["That's just how it is!", "You know what I mean?", "Classic situation!"]

        for i, tag in enumerate(tag_candidates[:3]):
            if i < 3:
                print(f"   • {tag}.")

        print("-" * 60)
        print("Raw output:", content[:200] + "..." if len(content) > 200 else content)
        print("-" * 60)
        return content

    def test_multiple_temperatures(self, prompt="Hey guys, welcome back to the podcast"):
        """Test generation with different temperature settings"""

        temperatures = [0.5, 0.7, 0.8, 0.9, 1.0]

        print("🔥 TEMPERATURE TESTING 🔥")
        print("=" * 60)

        for temp in temperatures:
            print(f"\n🌡️ Temperature: {temp}")
            print("-" * 30)
            self.generate_with_format(prompt, temperature=temp)
            print()

    def interactive_test(self):
        """Interactive testing mode"""

        print("🎪 Nick Mullen Comedy Tester 🎪")
        print("Type 'quit' to exit, 'temp' to test temperatures, or enter a prompt")
        print("-" * 50)

        while True:
            prompt = input("Enter prompt: ").strip()

            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'temp':
                test_prompt = input("Enter test prompt: ").strip() or "Hey guys, welcome back to the podcast"
                self.test_multiple_temperatures(test_prompt)
            elif prompt:
                self.generate_with_format(prompt)
                print()

def main():
    tester = ComedyFormatTester()

    # Quick test first
    print("\n🚀 QUICK TEST:")
    tester.generate_with_format("Hey guys, today's topic is dating disasters")

    print("\n" + "="*60)
    print("🎯 TESTING OPTIONS:")
    print("1. Interactive mode (recommended)")
    print("2. Temperature testing")
    print("3. Quick single test")

    choice = input("\nChoose (1-3): ").strip()

    if choice == '1':
        tester.interactive_test()
    elif choice == '2':
        prompt = input("Enter prompt for temperature testing: ").strip() or "Hey guys, welcome back to the podcast"
        tester.test_multiple_temperatures(prompt)
    elif choice == '3':
        prompt = input("Enter prompt: ").strip() or "Hey guys, welcome back to the podcast"
        tester.generate_with_format(prompt)
    else:
        print("Invalid choice, running quick test...")
        tester.generate_with_format("Tell me about your dating life")

if __name__ == "__main__":
    main()
