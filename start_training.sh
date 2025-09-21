#!/bin/bash

# Training startup script
# Handles PYTHONPATH and runs training

set -e

echo "🚀 starting nick mullen DialoGPT-small model training..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "src/training/train.py" ]; then
    echo -e "${RED}❌ error: run this script from the project root${NC}"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✅ activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}❌ no virtual environment found. run setup first.${NC}"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo -e "${GREEN}✅ set PYTHONPATH to $(pwd)${NC}"

# Test model access first
echo -e "${YELLOW}🧪 testing hugging face access...${NC}"
python src/training/test_model_download.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ hugging face access failed. check your token.${NC}"
    exit 1
fi

# Start training
echo -e "${GREEN}✅ starting training...${NC}"
python src/training/train.py

echo -e "${GREEN}🎉 training complete!${NC}"
