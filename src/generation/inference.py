#!/usr/bin/env python3
"""
Inference script for the fine-tuned Nick Mullen comedy model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NickMullenGenerator:
    def __init__(self, model_path="models/nick_mullen_model"):
        self.model_path = Path(model_path)
        self.generator = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        if not self.model_path.exists():
            logger.warning(f"Model path {self.model_path} does not exist. Using fallback.")
            return False

        try:
            logger.info(f"Loading model from {self.model_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
            )

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def generate_riff(self, topic, max_length=100):
        """Generate a comedy riff about a topic"""
        if not self.generator:
            raise Exception("Model not loaded - cannot generate riff")

        # More specific prompt for Nick Mullen style
        prompt = f"""You are Nick Mullen, the comedian from Cum Town and The Adam Friedland Show. Generate a short, edgy comedy riff about {topic}. Be absurd, ironic, and filthy like your style. Keep it under 50 words.

Example: "Imagine you're a guy who can't stop drawing swastikas on missing Israeli posters"

Riff about {topic}:"""

        try:
            result = self.generator(
                prompt,
                max_new_tokens=50,  # Shorter outputs
                num_return_sequences=1,
                temperature=1.2,  # Higher temperature for more creativity
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )

            generated_text = result[0]["generated_text"]
            # Extract just the generated part after the prompt
            riff = generated_text[len(prompt):].strip()

            # Clean up the riff
            riff = self._clean_riff(riff)

            # If riff is too long or empty, raise error
            if len(riff.split()) > 30 or len(riff.strip()) < 10:
                raise Exception(f"Generated riff too short or too long: {riff}")

            if not riff:
                raise Exception("Generated riff was empty")
            return riff

        except Exception as e:
            logger.error(f"Error generating riff: {e}")
            raise Exception(f"Failed to generate riff: {str(e)}")

    def generate_joke(self, topic, max_length=200):
        """Generate a structured joke about a topic"""
        if not self.generator:
            raise Exception("Model not loaded - cannot generate joke")

        prompt = f"""You are Nick Mullen from Cum Town. Generate a structured joke about {topic}:

Format as JSON:
{{
  "premise": "setup here",
  "punchline": "punchline here",
  "initial_tag": "tag here",
  "alternate_angle": "angle here",
  "additional_tags": ["tag1", "tag2"]
}}

JSON:"""

        try:
            result = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            generated_text = result[0]["generated_text"]

            # Try to extract JSON from the response
            import json
            import re

            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                try:
                    joke_data = json.loads(json_match.group())
                    return joke_data
                except json.JSONDecodeError:
                    pass

            # If JSON parsing fails, create structured response from text
            return self._parse_joke_from_text(generated_text)

        except Exception as e:
            logger.error(f"Error generating joke: {e}")
            raise Exception(f"Failed to generate joke: {str(e)}")

    def regenerate_joke_part(self, topic, joke_context, part_to_regenerate):
        """Regenerate a specific part of a joke"""
        if not self.generator:
            raise Exception("Model not loaded - cannot regenerate joke part")

        part_prompts = {
            "premise": f"Generate a new premise about {topic} in Nick Mullen's style:",
            "punchline": f"Generate a new punchline for this premise: {joke_context.get('premise', '')}",
            "initial_tag": f"Generate a new tag for this joke: {joke_context.get('premise', '')} -> {joke_context.get('punchline', '')}",
            "alternate_angle": f"Generate a new angle on this topic: {topic}",
            "additional_tags": f"Generate 2-3 new tags for this joke about {topic}"
        }

        if part_to_regenerate not in part_prompts:
            return joke_context.get(part_to_regenerate, "Generated content")

        prompt = f"""You are Nick Mullen from Cum Town. {part_prompts[part_to_regenerate]}

Style: Concise, punchy, absurd connections, raw phonetic sounds, modern slang when appropriate.
Keep it under 20 words.

Response:"""

        try:
            result = self.generator(
                prompt,
                max_length=50,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            new_content = result[0]["generated_text"].strip()

            # Clean up the generated content
            new_content = re.sub(r'^[^a-zA-Z0-9]*', '', new_content)
            new_content = re.sub(r'[^a-zA-Z0-9]*$', '', new_content)

            if part_to_regenerate == "additional_tags":
                # For tags, try to split into array format
                tags = [tag.strip() for tag in new_content.split(',') if tag.strip()]
                return tags[:3] if tags else ["Tag 1", "Tag 2"]

            return new_content

        except Exception as e:
            logger.error(f"Error regenerating joke part: {e}")
            raise Exception(f"Failed to regenerate joke part: {str(e)}")

    def _clean_riff(self, riff):
        """Clean up generated riff text"""
        # Remove extra whitespace and newlines
        riff = ' '.join(riff.split())

        # Remove any incomplete sentences at the end
        if riff and not riff[-1] in '.!?':
            # Find last complete sentence
            sentences = riff.split('.')
            if len(sentences) > 1:
                riff = '.'.join(sentences[:-1]) + '.'

        return riff.strip()

    def _parse_joke_from_text(self, text):
        """Parse joke structure from generated text if JSON fails"""
        lines = text.split('\n')
        joke_parts = {
            "premise": "",
            "punchline": "",
            "initial_tag": "",
            "alternate_angle": "",
            "additional_tags": []
        }

        current_part = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to identify different parts
            if "premise" in line.lower():
                current_part = "premise"
                joke_parts["premise"] = line.split(":", 1)[-1].strip()
            elif "punchline" in line.lower():
                current_part = "punchline"
                joke_parts["punchline"] = line.split(":", 1)[-1].strip()
            elif "tag" in line.lower():
                if not joke_parts["initial_tag"]:
                    current_part = "initial_tag"
                    joke_parts["initial_tag"] = line.split(":", 1)[-1].strip()
                else:
                    joke_parts["additional_tags"].append(line.split(":", 1)[-1].strip())
            elif "angle" in line.lower():
                current_part = "alternate_angle"
                joke_parts["alternate_angle"] = line.split(":", 1)[-1].strip()

        return joke_parts


# Global generator instance
_generator = None

def get_generator():
    """Get or create the global generator instance"""
    global _generator
    if _generator is None:
        _generator = NickMullenGenerator()
    return _generator

def generate_riff(topic):
    """Convenience function for riff generation"""
    return get_generator().generate_riff(topic)

def generate_joke(topic):
    """Convenience function for joke generation"""
    return get_generator().generate_joke(topic)

def regenerate_joke_part(topic, joke_context, part_to_regenerate):
    """Convenience function for regenerating joke parts"""
    return get_generator().regenerate_joke_part(topic, joke_context, part_to_regenerate)

if __name__ == "__main__":
    # Test the generator
    generator = get_generator()

    print("Testing riff generation:")
    riff = generator.generate_riff("dating apps")
    print(f"Riff: {riff}")

    print("\nTesting joke generation:")
    joke = generator.generate_joke("politics")
    print(f"Joke: {joke}")
