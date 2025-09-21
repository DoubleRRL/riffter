#!/bin/bash

# Riffter startup script
# Activates venv, checks deps, starts backend and frontend

set -e  # Exit on any error

echo "🚀 starting riffter..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -d "frontend" ]; then
    echo -e "${RED}❌ error: run this script from the project root${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  no virtual environment found. creating one...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}✅ activating virtual environment...${NC}"
source venv/bin/activate

# Install/update dependencies
echo -e "${GREEN}✅ installing python dependencies...${NC}"
pip install -q -r requirements.txt

# Check if dependencies are properly installed
echo -e "${GREEN}✅ checking python dependencies...${NC}"
python -c "import fastapi, uvicorn, transformers, torch, ollama; print('✅ all deps good')" || {
    echo -e "${RED}❌ python dependencies check failed${NC}"
    exit 1
}

# Install frontend dependencies
echo -e "${GREEN}✅ installing frontend dependencies...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
else
    echo "📦 node_modules already exists, skipping npm install"
fi
cd ..

# Start backend in background
echo -e "${GREEN}✅ starting backend on port 8000...${NC}"
PYTHONPATH="/Users/RRL_1/riffter:$PYTHONPATH" python src/api/main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ backend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✅ backend running on http://localhost:8000${NC}"

# Start frontend
echo -e "${GREEN}✅ starting frontend...${NC}"
cd frontend

# Get available port for frontend
PORT=5173
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    echo "📍 port $PORT is busy, trying $((PORT + 1))"
    PORT=$((PORT + 1))
done

echo -e "${GREEN}✅ frontend will run on port $PORT${NC}"

# Start frontend in background
npm run dev -- --port $PORT &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}❌ frontend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✅ frontend running on http://localhost:$PORT${NC}"

# Update CORS origins in backend if needed
if [ "$PORT" != "5173" ] && [ "$PORT" != "3000" ]; then
    echo -e "${YELLOW}⚠️  frontend running on non-standard port $PORT${NC}"
    echo -e "${YELLOW}⚠️  you may need to update CORS origins in src/api/main.py${NC}"
    echo -e "${YELLOW}⚠️  add 'http://localhost:$PORT' to the allow_origins list${NC}"
fi

echo ""
echo -e "${GREEN}🎉 riffter is running!${NC}"
echo -e "${GREEN}📡 backend: http://localhost:8000${NC}"
echo -e "${GREEN}🌐 frontend: http://localhost:$PORT${NC}"
echo ""
echo -e "${YELLOW}💡 tip: if you get CORS errors, add http://localhost:$PORT to the allow_origins in src/api/main.py${NC}"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user interrupt
trap "echo '🛑 stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
