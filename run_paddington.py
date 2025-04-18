#!/usr/bin/env python3
"""
Paddington Voice Assistant - Simplified Runner
----------------------------------------------
This script installs the required dependencies and starts the Paddington server.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed and install them if needed"""
    print("Checking dependencies...")
    
    # Check if Zonos is properly installed
    try:
        # Try importing zonos
        try:
            import zonos
            print("✓ Zonos is already installed")
        except ImportError:
            print("Zonos is not installed. Let's install it...")
            
            # Install required dependencies for Zonos
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=2.2.2"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile>=0.13.1"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.5.1", "torchaudio>=2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.48.1"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "sentencepiece", "protobuf"])
            
            # Install Zonos from GitHub
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Zyphra/Zonos.git"])
            print("✓ Zonos installed successfully")
            
        # Check for other required packages
        required_packages = [
            "fastapi", "uvicorn", "python-multipart", "whisper", 
            "llama-cpp-python", "websockets", "python-dotenv"
        ]
        
        # Use pip to check and install missing packages
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✓ {package} is already installed")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
        
        print("All dependencies are installed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    
def create_directories():
    """Create necessary directories if they don't exist"""
    print("Setting up directory structure...")
    
    # Get the base directory (where this script is located)
    base_dir = Path(__file__).parent.absolute()
    
    # Create required directories
    directories = [
        base_dir / "src" / "static" / "audio",
        base_dir / "src" / "uploads",
        base_dir / "src" / "temp",
        base_dir / "data",
        base_dir / "src" / "assets",
    ]
    
    for directory in directories:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    # Create example audio file for voice cloning if it doesn't exist
    example_audio_path = base_dir / "src" / "assets" / "exampleaudio.mp3"
    if not example_audio_path.exists():
        print("Note: No example audio found for voice cloning.")
        print("You may want to add a sample voice file at: src/assets/exampleaudio.mp3")
    
    print("Directory structure is ready!")

def create_dotenv():
    """Create a .env file if it doesn't exist"""
    env_file = Path(__file__).parent.absolute() / ".env"
    
    if not env_file.exists():
        print("Creating default .env file...")
        
        with open(env_file, "w") as f:
            f.write("""# Paddington Environment Variables
ZONOS_MODEL_PATH=/data/models/zonos_hybrid/model.safetensors
ZONOS_CONFIG_PATH=/data/models/zonos_hybrid/config.json
LLM_MODEL_PATH=/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
PERSONALITY_PATH=data/personality.json
USE_SSL=False
""")
        print("Created default .env file")
    else:
        print(".env file already exists")

def start_server():
    """Start the Paddington server"""
    print("\n" + "=" * 50)
    print("Starting Paddington Voice Assistant Server")
    print("=" * 50 + "\n")
    
    # Get the base directory
    base_dir = Path(__file__).parent.absolute()
    
    # Change to the src directory
    os.chdir(base_dir / "src")
    
    try:
        # Start the API server
        subprocess.run([sys.executable, "api.py"], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer stopped with error: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Paddington Voice Assistant Setup")
    print("=" * 50 + "\n")
    
    # Setup steps
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\nError: Failed to install dependencies. Please run update_zonos_deps.sh manually.")
        sys.exit(1)
        
    create_directories()
    create_dotenv()
    
    print("\nStarting server... If you encounter any errors related to Zonos,")
    print("please run the update_zonos_deps.sh script first.")
    start_server() 