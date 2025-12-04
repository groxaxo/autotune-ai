#!/bin/bash
# Quick start script for Autotune-AI web frontend

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Autotune-AI Web Frontend Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python is not installed${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
echo -e "${GREEN}Checking dependencies...${NC}"
if ! $PYTHON_CMD -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p frontend/uploads frontend/outputs

# Check for GPU
echo -e "${GREEN}Checking for GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo -e "${GREEN}GPU detected and available${NC}"
else
    echo -e "${YELLOW}No GPU detected. Will use CPU (slower processing)${NC}"
fi

# Start the server
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting Autotune-AI web server...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Access the web interface at: ${GREEN}http://localhost:5000${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop the server"
echo ""

cd frontend
$PYTHON_CMD app.py
