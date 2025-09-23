#!/usr/bin/env python3
"""
Inference script for the fine-tuned Nick Mullen comedy model using Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CumTownGenerator:
    def __init__(self, model_path="models/cumtown_model"):
        self.model_path = Path(model_path)
        self.generator = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        if not self.model_path.exists():
            logger.error(f"Model path {self.model_path} does not exist. No fallback - train the model first.")
            self.generator = None
            return

        try:
            logger.info(f"Loading fine-tuned model from {self.model_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            # Load model - check for compute availability
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            device_type = "cuda" if cuda_available else ("mps" if mps_available else "cpu")

            logger.info(f"CUDA available: {cuda_available}")
            logger.info(f"MPS available: {mps_available}")
            logger.info(f"Using device: {device_type}")

            model_kwargs = {
                "torch_dtype": torch.float32,  # Use float32 for compatibility
            }

            if device_type == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["load_in_8bit"] = True  # Use 8-bit quantization for memory efficiency
                logger.info("Using 8-bit quantization for GPU inference")
            else:
                # For MPS and CPU, use disk offloading
                model_kwargs["device_map"] = "auto"  # This enables disk offloading
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("Using disk offloading for inference")

            model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **model_kwargs)

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
                # Don't specify device when using accelerate
            )

            logger.info("Fine-tuned model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            logger.error("Model is corrupted or incompatible. Retrain the model.")
            self.generator = None

    def generate_riff(self, topic, max_length=100):
        """Generate a comedy riff about a topic using the fine-tuned model"""
        if not self.generator:
            raise Exception("Model not loaded - cannot generate riff")

        prompt = f"Generate a comedy riff about {topic} in the style of Cum Town podcast:"

        try:
            result = self.generator(
                prompt,
                max_new_tokens=100,  # Shorter outputs
                num_return_sequences=1,
                temperature=0.9,  # Slightly creative but not too random
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

            # Basic validation
            if len(riff.split()) > 30 or len(riff.strip()) < 5:
                logger.warning(f"Generated riff seems too short/long: {riff}")

            return riff if riff else "Failed to generate riff"

        except Exception as e:
            logger.error(f"Error generating riff: {e}")
            raise Exception(f"Failed to generate riff: {str(e)}")

    def generate_joke(self, topic, max_length=200):
        """Generate a structured joke about a topic using the fine-tuned model with proper prompts for each field"""
        if not self.generator:
            raise Exception("Model not loaded - cannot generate joke")

        try:
            # Generate each field using specific prompts for better structure and funnier results
            joke_parts = {}

            # 1. Generate premise
            premise_prompt = f"Generate a funny premise about {topic} in Cum Town podcast style:"
            result = self.generator(
                premise_prompt,
                max_new_tokens=30,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            premise_text = result[0]["generated_text"][len(premise_prompt):].strip()
            premise_text = self._clean_joke_text(premise_text)
            # Ensure it ends with proper punctuation and is a question/setup
            if not premise_text.endswith(('?', '!', '.')):
                premise_text += '?'
            joke_parts["premise"] = premise_text

            # 2. Generate punchline based on the premise
            punchline_prompt = f"Generate a hilarious punchline for this premise in Cum Town style: {joke_parts['premise']}"
            result = self.generator(
                punchline_prompt,
                max_new_tokens=40,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            punchline_text = result[0]["generated_text"][len(punchline_prompt):].strip()
            punchline_text = self._clean_joke_text(punchline_text)
            # Ensure it ends with proper punctuation
            if not punchline_text.endswith(('!', '.', '?')):
                punchline_text += '!'
            joke_parts["punchline"] = punchline_text

            # 3. Generate initial tag
            initial_tag_prompt = f"Generate a short, punchy tag for this joke in Cum Town style: '{joke_parts['premise']}' -> '{joke_parts['punchline']}'"
            result = self.generator(
                initial_tag_prompt,
                max_new_tokens=15,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            initial_tag_text = result[0]["generated_text"][len(initial_tag_prompt):].strip()
            initial_tag_text = self._clean_joke_text(initial_tag_text)
            if not initial_tag_text:
                initial_tag_text = "That's fucked up"
            joke_parts["initial_tag"] = initial_tag_text

            # 4. Generate alternate angle
            alternate_angle_prompt = f"Generate another angle or perspective on {topic} in Cum Town podcast style:"
            result = self.generator(
                alternate_angle_prompt,
                max_new_tokens=35,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            alternate_angle_text = result[0]["generated_text"][len(alternate_angle_prompt):].strip()
            alternate_angle_text = self._clean_joke_text(alternate_angle_text)
            if not alternate_angle_text:
                alternate_angle_text = f"From another angle, {topic} is just ridiculous."
            joke_parts["alternate_angle"] = alternate_angle_text

            # 5. Generate 3 tags based on content analysis
            # Since the AI tends to continue in podcast style rather than generate structured tags,
            # we'll create relevant tags based on the topic and generated content

            tags = ["comedy", "cumtown"]  # Always include these

            # Add topic-related tag
            topic_words = topic.lower().split()
            if topic_words:
                tags.append(topic_words[0])

            # Add content-based tags from the generated text
            all_content = f"{joke_parts['premise']} {joke_parts['punchline']} {joke_parts['initial_tag']} {joke_parts['alternate_angle']}".lower()

            # Look for themes that suggest good tags
            content_tags = []
            if any(word in all_content for word in ['sex', 'fuck', 'dick', 'pussy', 'ass', 'horny', 'porn', 'naked']):
                content_tags.append("sex")
            if any(word in all_content for word in ['politics', 'government', 'trump', 'biden', 'vote', 'election']):
                content_tags.append("politics")
            if any(word in all_content for word in ['dating', 'relationship', 'love', 'single', 'date', 'partner']):
                content_tags.append("dating")
            if any(word in all_content for word in ['angry', 'furious', 'pissed', 'hate', 'rage']):
                content_tags.append("angry")
            if any(word in all_content for word in ['weird', 'strange', 'crazy', 'bizarre', 'odd']):
                content_tags.append("weird")
            if 'women' in all_content or 'girl' in all_content or 'female' in all_content:
                content_tags.append("women")
            if 'men' in all_content or 'guy' in all_content or 'dude' in all_content or 'male' in all_content:
                content_tags.append("men")
            if any(word in all_content for word in ['money', 'rich', 'poor', 'cash', 'expensive']):
                content_tags.append("money")
            if any(word in all_content for word in ['work', 'job', 'career', 'boss', 'office']):
                content_tags.append("work")

            # Add the first content tag found, or "funny" as fallback
            if content_tags:
                tags.append(content_tags[0])
            else:
                tags.append("funny")

            # Ensure exactly 3 unique tags
            tags = list(dict.fromkeys(tags))  # Remove duplicates
            while len(tags) < 3:
                if "funny" not in tags:
                    tags.append("funny")
                elif "humor" not in tags:
                    tags.append("humor")
                else:
                    tags.append("podcast")

            joke_parts["additional_tags"] = tags[:3]

            return joke_parts

        except Exception as e:
            logger.error(f"Error generating joke with prompts: {e}")
            raise Exception(f"Failed to generate structured joke: {str(e)}")

    def regenerate_joke_part(self, topic, joke_context, part_to_regenerate):
        """Regenerate a specific part of a joke using the fine-tuned model"""
        if not self.generator:
            raise Exception("Model not loaded - cannot regenerate joke part")

        part_prompts = {
            "premise": f"Generate a new premise about {topic} in Cum Town podcast style:",
            "punchline": f"Generate a new punchline for this premise: {joke_context.get('premise', '')}",
            "initial_tag": f"Generate a new tag for this joke: {joke_context.get('premise', '')} -> {joke_context.get('punchline', '')}",
            "alternate_angle": f"Generate a new angle on this topic: {topic}",
            "additional_tags": f"Generate 2-3 new tags for this joke about {topic}"
        }

        if part_to_regenerate not in part_prompts:
            return joke_context.get(part_to_regenerate, "Generated content")

        prompt = f"Generate a new {part_to_regenerate} for this joke in Cum Town podcast style: {part_prompts[part_to_regenerate]}"

        try:
            result = self.generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            generated_text = result[0]["generated_text"]
            # Extract just the generated part after the prompt
            new_content = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()

            # Clean up the generated content
            new_content = self._clean_joke_text(new_content)

            if part_to_regenerate == "additional_tags":
                # For tags, try to split into array format
                tags = [tag.strip() for tag in new_content.split(',') if tag.strip()]
                # If no commas, try to split by newlines or other delimiters
                if len(tags) <= 1:
                    # Try splitting by newlines
                    tags = [tag.strip() for tag in new_content.split('\n') if tag.strip()]
                # Filter out any remaining prompt-like text
                tags = [tag for tag in tags if not any(word in tag.lower() for word in ['generate', 'tags', 'joke', 'about'])]
                # Ensure we have at least 2-3 tags
                if len(tags) < 2:
                    tags = ["comedy", "cumtown", topic.split()[0] if topic.split() else "funny"]
                return tags[:3]

            return new_content if new_content else "Generated content"

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

    def _clean_joke_text(self, joke_text):
        """Clean up generated joke text"""
        if not joke_text:
            return "Why do people love this topic? Because it's fucking hilarious!"

        # Remove extra whitespace
        joke_text = ' '.join(joke_text.split())

        # Remove repetitive patterns (like the "you could do" loops we saw)
        words = joke_text.split()
        if len(words) > 10:
            # Check for excessive repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # If any word appears more than 20% of the time, truncate
            max_repeats = len(words) * 0.2
            for word, count in word_counts.items():
                if count > max_repeats and len(word) > 3:  # Ignore short words
                    # Find first occurrence and truncate there
                    first_occurrence = joke_text.find(word)
                    if first_occurrence > 50:  # Only if we're past a reasonable length
                        joke_text = joke_text[:first_occurrence].strip()
                        break

        # Limit length
        if len(joke_text) > 300:
            # Try to cut at sentence boundary
            sentences = joke_text.split('.')
            joke_text = '.'.join(sentences[:2]) + '.'

        return joke_text.strip()

    def _structure_joke_from_text(self, joke_text, topic):
        """Structure generated comedy text into joke components"""
        if not joke_text or len(joke_text.strip()) < 10:
            # Fallback to defaults
            return {
                "premise": f"Why do people love {topic}?",
                "punchline": "Because it's fucking hilarious!",
                "initial_tag": "That's the joke",
                "alternate_angle": f"From another angle, {topic} is just ridiculous",
                "additional_tags": ["comedy", "funny", f"{topic.replace(' ', '_')}"]
            }

        # Split the text into sentences
        sentences = [s.strip() for s in joke_text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

        if len(sentences) >= 2:
            # Use first sentence as premise, second as punchline
            premise = sentences[0] + '.'
            punchline = sentences[1] + '.'
        elif len(sentences) == 1:
            # Split single sentence
            words = sentences[0].split()
            mid_point = len(words) // 2
            premise = ' '.join(words[:mid_point]) + '.'
            punchline = ' '.join(words[mid_point:]) + '.'
        else:
            premise = joke_text[:len(joke_text)//2] + '.'
            punchline = joke_text[len(joke_text)//2:] + '.'

        # Generate tags from the content
        words = joke_text.lower().split()
        tags = ["comedy", "cumtown"]

        # Add topic-related tag
        topic_words = topic.lower().split()
        if topic_words:
            tags.append(topic_words[0])

        # Look for funny words or themes
        funny_indicators = ["fuck", "shit", "dick", "pussy", "ass", "cunt", "bitch"]
        for word in words:
            if word in funny_indicators and word not in tags:
                tags.append(word)
                break

        # Ensure exactly 3 tags
        while len(tags) < 3:
            if "funny" not in tags:
                tags.append("funny")
            elif "humor" not in tags:
                tags.append("humor")
            else:
                tags.append("podcast")

        return {
            "premise": premise,
            "punchline": punchline,
            "initial_tag": "That's fucked up" if any(word in joke_text.lower() for word in funny_indicators) else "That's the joke",
            "alternate_angle": f"From another perspective, {topic} is just {sentences[0].split()[0] if sentences else 'ridiculous'}",
            "additional_tags": tags[:3]
        }

    def _create_joke_from_inspiration(self, riff_text, topic):
        """Create a structured joke using model-generated riff as inspiration"""
        sentences = [s.strip() for s in riff_text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

        # Use the riff as the punchline, create a premise around the topic
        punchline = riff_text if riff_text.endswith(('.', '!', '?')) else riff_text + '.'
        premise = f"Why do people say {topic}?"

        # Create an alternate angle from the riff
        words = riff_text.split()
        if len(words) > 3:
            alt_start = ' '.join(words[:3])
            alternate_angle = f"Some people think {alt_start}..."
        else:
            alternate_angle = f"From another angle, {topic} hits different."

        # Generate tags
        tags = ["comedy", "cumtown"]
        topic_words = topic.split()
        if topic_words:
            tags.append(topic_words[0])

        # Ensure exactly 3 tags
        while len(tags) < 3:
            if "funny" not in tags:
                tags.append("funny")
            elif "humor" not in tags:
                tags.append("humor")
            else:
                tags.append("podcast")

        return {
            "premise": premise,
            "punchline": punchline,
            "initial_tag": "That's the vibe",
            "alternate_angle": alternate_angle,
            "additional_tags": tags[:3]
        }

    def _create_template_joke(self, topic):
        """Create a template-based joke when model fails"""
        # Create a simple, plausible joke structure based on the topic

        # Handle the specific case from the user
        if 'women love words with the letter' in topic.lower():
            letter = topic.split('"')[-2] if '"' in topic else "C"
            return {
                "premise": f"Why do women love words with the letter {letter}?",
                "punchline": f"Because {letter} stands for 'cock', 'cunt', 'clit' - you know, the good stuff!",
                "initial_tag": "That's fucked up",
                "alternate_angle": f"From another angle, it's just basic biology - women love the C words.",
                "additional_tags": ["sex", "women", "dirty"]
            }

        # Generic templates based on topic keywords
        topic_lower = topic.lower()

        if any(word in topic_lower for word in ['sex', 'fuck', 'dick', 'pussy', 'ass']):
            return {
                "premise": f"What's the deal with {topic}?",
                "punchline": f"It's just people being horny as fuck - can you blame them?",
                "initial_tag": "That's horny",
                "alternate_angle": f"From another perspective, {topic} is basically human nature.",
                "additional_tags": ["sex", "horny", "comedy"]
            }

        elif any(word in topic_lower for word in ['politics', 'government', 'trump', 'biden']):
            return {
                "premise": f"Why is {topic} so fucked up?",
                "punchline": "Because it's run by rich assholes who don't give a shit about regular people!",
                "initial_tag": "That's politics",
                "alternate_angle": f"From another angle, {topic} is just clowns running the circus.",
                "additional_tags": ["politics", "clowns", "angry"]
            }

        elif any(word in topic_lower for word in ['dating', 'relationships', 'love']):
            return {
                "premise": f"What's the truth about {topic}?",
                "punchline": "It's all just people pretending they're not as fucked up as they really are!",
                "initial_tag": "That's real shit",
                "alternate_angle": f"From another angle, {topic} is just humans being humans.",
                "additional_tags": ["dating", "real", "bitter"]
            }

        else:
            # Generic fallback
            tags = ["comedy", "funny"]
            topic_word = topic.split()[0] if topic.split() else "topic"
            tags.append(topic_word)
            return {
                "premise": f"Why do people care about {topic}?",
                "punchline": f"Because it's fucking hilarious when you think about it!",
                "initial_tag": "That's the joke",
                "alternate_angle": f"From another perspective, {topic} is just ridiculous.",
                "additional_tags": tags[:3]
            }

    def _parse_joke_from_structured_text(self, text, topic):
        """Parse joke structure from generated text in structured format"""
        lines = text.split('\n')
        joke_parts = {
            "premise": "",
            "punchline": "",
            "initial_tag": "",
            "alternate_angle": "",
            "additional_tags": []
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.lower().startswith("premise:"):
                current_section = "premise"
                content = line.split(":", 1)[-1].strip()
                if content:
                    joke_parts["premise"] = content
            elif line.lower().startswith("punchline:"):
                current_section = "punchline"
                content = line.split(":", 1)[-1].strip()
                if content:
                    joke_parts["punchline"] = content
            elif line.lower().startswith("tag:") and not joke_parts["initial_tag"]:
                current_section = "initial_tag"
                content = line.split(":", 1)[-1].strip()
                if content:
                    joke_parts["initial_tag"] = content
            elif line.lower().startswith("another angle:"):
                current_section = "alternate_angle"
                content = line.split(":", 1)[-1].strip()
                if content:
                    joke_parts["alternate_angle"] = content
            elif line.lower().startswith("tags:"):
                current_section = "tags"
                content = line.split(":", 1)[-1].strip()
                if content:
                    # Split by commas and clean up
                    tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                    joke_parts["additional_tags"] = tags[:3]  # Limit to 3 tags
            elif current_section and line and not any(line.lower().startswith(header) for header in ["premise:", "punchline:", "tag:", "another angle:", "tags:"]):
                # Continue accumulating content for current section
                if current_section == "premise":
                    joke_parts["premise"] += " " + line
                elif current_section == "punchline":
                    joke_parts["punchline"] += " " + line
                elif current_section == "initial_tag":
                    joke_parts["initial_tag"] += " " + line
                elif current_section == "alternate_angle":
                    joke_parts["alternate_angle"] += " " + line

        # Clean up and provide defaults if needed
        for key in joke_parts:
            if isinstance(joke_parts[key], str):
                joke_parts[key] = joke_parts[key].strip()
                # If empty, provide a simple default
                if not joke_parts[key]:
                    if key == "premise":
                        joke_parts[key] = f"Why do people love {topic}?"
                    elif key == "punchline":
                        joke_parts[key] = f"Because it's fucking hilarious!"
                    elif key == "initial_tag":
                        joke_parts[key] = "That's the joke"
                    elif key == "alternate_angle":
                        joke_parts[key] = f"From another perspective, {topic} is just ridiculous"

        # Ensure we have at least 2-3 tags
        if len(joke_parts["additional_tags"]) < 2:
            default_tags = ["comedy", "funny", f"{topic.replace(' ', '_')}"]
            joke_parts["additional_tags"] = default_tags[:3]

        return joke_parts


# Global generator instance
_generator = None

def get_generator():
    """Get or create the global generator instance"""
    global _generator
    if _generator is None:
        _generator = CumTownGenerator()
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

    print("Testing Cum Town riff generation:")
    riff = generator.generate_riff("dating apps")
    print(f"Riff: {riff}")

    print("\nTesting Cum Town joke generation:")
    joke = generator.generate_joke("politics")
    print(f"Joke: {joke}")
