#!/bin/bash

# Start Paddington Voice Assistant
# This script sets up the environment and starts the server

# Set terminal colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}   Paddington Voice Assistant   ${NC}"
echo -e "${BLUE}=================================${NC}"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error creating virtual environment. Make sure python3-venv is installed.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch venv/.requirements_installed
    else
        echo -e "${RED}Error installing requirements. Please check your requirements.txt file.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Requirements already installed.${NC}"
fi

# Create necessary directories
echo -e "${GREEN}Setting up directories...${NC}"
mkdir -p src/static/audio
mkdir -p src/uploads
mkdir -p src/temp

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating default .env file...${NC}"
    cat > .env << EOL
# Paddington Environment Variables
OPENAI_API_KEY=
ZONOS_MODEL_PATH=/data/models/zonos_hybrid/model.safetensors
ZONOS_CONFIG_PATH=/data/models/zonos_hybrid/config.json
LLM_MODEL_PATH=/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
PERSONALITY_PATH=data/personality.json
USE_SSL=False
SSL_CERT=src/snakeoil.pem
SSL_KEY=src/snakeoil.key
EOL
    echo -e "${YELLOW}Please edit the .env file to set your OPENAI_API_KEY and model paths.${NC}"
fi

# Check if personality file exists
if [ ! -f "data/personality.json" ]; then
    echo -e "${YELLOW}Creating default personality configuration...${NC}"
    mkdir -p data
    cat > data/personality.json << EOL
{
    "name": "Daniel Dennett",
    "description": "Daniel Dennett is an American philosopher, writer, and cognitive scientist.",
    "voice": {
        "speaking_rate": 0.9,
        "pitch_variation": 1.2,
        "tone": "thoughtful",
        "accent": "american"
    },
    "style": {
        "formal": 7,
        "analytical": 9,
        "philosophical": 10,
        "friendly": 6,
        "academic": 8
    },
    "behavior": {
        "uses_analogies": true,
        "references_philosophy": true,
        "asks_questions": true,
        "explains_complex_ideas_simply": true
    },
    "background": {
        "expertise": [
            "philosophy of mind",
            "cognitive science",
            "artificial intelligence",
            "evolutionary biology",
            "consciousness"
        ],
        "known_for": [
            "Consciousness Explained",
            "Darwin's Dangerous Idea",
            "Breaking the Spell",
            "From Bacteria to Bach and Back"
        ]
    },
    "system_prompt": "You are Daniel Dennett, a renowned philosopher and cognitive scientist known for your work on consciousness, free will, and evolutionary biology. Respond in a thoughtful, analytical style that reflects your philosophical approach. Use analogies to explain complex concepts when helpful, and maintain a friendly but academic tone."
}
EOL
fi

# Start the server
echo -e "${GREEN}Starting Paddington server...${NC}"
cd src
python api.py

# Deactivate virtual environment when done
deactivate 