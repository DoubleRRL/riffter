#!/usr/bin/env python3
"""
Quick test training script with just a few samples to verify everything works
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_dataset():
    """Create a small test dataset from the Cum Town data"""
    data_path = "data/cumtown_training_data.json"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Take just the first 5 samples for testing (even smaller for CPU)
    test_data = data[:5]

    test_data_path = "data/test_training_data.json"
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=2)

    logger.info(f"Created test dataset with {len(test_data)} samples at {test_data_path}")
    return test_data_path

if __name__ == "__main__":
    test_data_path = create_test_dataset()

    print("\n" + "="*60)
    print("TEST TRAINING SETUP")
    print("="*60)
    print(f"Test data: {test_data_path}")
    print()
    print("To test training with a small dataset:")
    print("python src/training/quick_test.py")
    print()
    print("This will train on just 5 samples for 1 step to verify everything works")
    print("before committing to training on all 31k samples")
    print("="*60)
