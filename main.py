#!/usr/bin/env python3
"""
Main entry point for Riffter - Comedy AI Pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the API server
from api.main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Riffter API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
