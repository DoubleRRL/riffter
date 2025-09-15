#!/usr/bin/env python3
"""
Test loading the Llama 3.1-8B model in Meta format
"""

import torch
from pathlib import Path
import json
from transformers import LlamaTokenizer

def test_model_loading():
    """Test loading the downloaded Llama model"""

    model_path = Path("/Users/RRL_1/.llama/checkpoints/Llama3.1-8B")

    print("🔍 TESTING LLAMA 3.1-8B MODEL LOADING")
    print("=" * 50)

    try:
        # Check if files exist
        required_files = ['params.json', 'tokenizer.model', 'consolidated.00.pth']
        for file in required_files:
            file_path = model_path / file
            if file_path.exists():
                print(f"✅ Found: {file}")
            else:
                print(f"❌ Missing: {file}")
                return False

        # Load model parameters
        print("\n📋 Loading model parameters...")
        with open(model_path / "params.json", 'r') as f:
            params = json.load(f)
        print(f"Model config: {params}")

        # Try to load tokenizer
        print("\n🔤 Loading tokenizer...")
        try:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(str(model_path), legacy=True)
            print("✅ Tokenizer loaded successfully")

            # Test tokenization
            test_text = "Hey guys, welcome back to the podcast"
            tokens = tokenizer(test_text, return_tensors="pt")['input_ids']
            print(f"Test text: '{test_text}'")
            print(f"Tokens: {tokens[0][:10].tolist()}... (showing first 10)")

        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
            return False

        # Try to load model weights
        print("\n🏋️ Testing model weights loading...")
        try:
            checkpoint = torch.load(model_path / "consolidated.00.pth", map_location="cpu")
            print(f"✅ Checkpoint loaded: {len(checkpoint)} parameters")
            print(".1f")

            # Show some parameter names
            param_names = list(checkpoint.keys())[:5]
            print(f"Sample parameters: {param_names}")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

        print("\n🎉 MODEL LOADING TEST PASSED!")
        print("The model can be loaded successfully.")
        print("\nNext steps:")
        print("1. Convert to Hugging Face format for easier use")
        print("2. Or create custom loading for Meta format")
        print("3. Test generation with the model")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)
