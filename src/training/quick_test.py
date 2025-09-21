#!/usr/bin/env python3
"""
Quick training test with minimal data to verify everything works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.train import train_model

if __name__ == "__main__":
    print("🧪 Running quick training test...")

    # Train on test data with minimal settings
    train_model(
        data_path="data/test_training_data.json",
        epochs=1,  # Just 1 epoch
        batch_size=1,
        learning_rate=2e-5,
        gradient_accumulation_steps=1  # Minimal accumulation
    )

    print("✅ Training test completed successfully!")
    print("🎉 You can now run the full training: ./start_training.sh")
