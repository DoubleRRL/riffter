#!/usr/bin/env python3
"""
Create Ollama Modelfile for fine-tuning on Cum Town data
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ollama_modelfile(training_data_path: str, output_dir: str = "ollama_model"):
    """Create Ollama Modelfile and training data"""

    # Load processed training data
    with open(training_data_path, 'r') as f:
        data = json.load(f)

    # Extract just the text content
    texts = [item["text"] for item in data if "text" in item]

    logger.info(f"Loaded {len(texts)} training samples")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create training data file for Ollama
    training_file = output_path / "training_data.txt"
    with open(training_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n\n")  # Double newline separates examples

    # Create Modelfile
    modelfile_content = f"""FROM llama3-lexi-uncensored

# Set parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for Nick Mullen style
SYSTEM \"\"\"You are Nick Mullen from Cum Town. Generate comedy in the style of Nick Mullen - raw, edgy, absurd, with phonetic sounds, modern slang, and unfiltered observations. Be funny like the podcast, not corporate-clean.\"\"\"

# Training data
ADAPTER {training_file}
"""

    modelfile_path = output_path / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"Created Ollama files in {output_dir}")
    logger.info(f"Modelfile: {modelfile_path}")
    logger.info(f"Training data: {training_file}")

    return str(modelfile_path)

def main():
    training_data_path = "data/cumtown_training_data.json"

    if not Path(training_data_path).exists():
        logger.error(f"Training data not found: {training_data_path}")
        logger.error("Run: python src/training/process_cumtown_data.py")
        return

    # Create Ollama files
    modelfile_path = create_ollama_modelfile(training_data_path)

    print("\n" + "="*50)
    print("OLLAMA FINE-TUNING SETUP COMPLETE")
    print("="*50)
    print(f"Files created in: ollama_model/")
    print()
    print("TO TRAIN YOUR MODEL:")
    print("1. cd ollama_model")
    print("2. ollama create nick-mullen-cumtown -f Modelfile")
    print("3. Wait for training to complete (this takes time)")
    print("4. Test with: ollama run nick-mullen-cumtown")
    print()
    print("The model will be available as 'nick-mullen-cumtown' in Ollama")

if __name__ == "__main__":
    main()
