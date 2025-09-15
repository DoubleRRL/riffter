#!/usr/bin/env python3
"""
Simple test to verify Llama model files are accessible
"""

from pathlib import Path

def verify_model_files():
    """Verify model files exist and are accessible"""

    model_path = Path("/Users/RRL_1/.llama/checkpoints/Llama3.1-8B")

    print("🔍 VERIFYING LLAMA 3.1-8B MODEL FILES")
    print("=" * 45)

    files = {
        'params.json': 'Model configuration',
        'tokenizer.model': 'SentencePiece tokenizer',
        'consolidated.00.pth': 'Model weights (16GB+)',
        'checklist.chk': 'Download verification'
    }

    all_good = True
    for filename, description in files.items():
        file_path = model_path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            if size > 1024**3:  # GB
                size_str = ".1f"
            elif size > 1024**2:  # MB
                size_str = ".1f"
            else:
                size_str = f"{size} bytes"
            print(f"✅ {filename} ({size_str}) - {description}")
        else:
            print(f"❌ {filename} - MISSING - {description}")
            all_good = False

    print("\n📊 SUMMARY:")
    if all_good:
        print("🎉 All model files are present and ready!")
        print("Next: Fine-tune the model on comedy data")
    else:
        print("⚠️ Some files are missing - re-download may be needed")

    return all_good

if __name__ == "__main__":
    verify_model_files()
