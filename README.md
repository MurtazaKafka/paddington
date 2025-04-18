# Paddington Voice Assistant

A sophisticated AI voice assistant that combines Large Language Models (LLM) with Text-to-Speech (TTS) to create an interactive voice-based conversation experience. Paddington is designed to mimic philosopher Daniel Dennett's voice and conversational style, delivering thoughtful responses on philosophy, consciousness, and a range of topics.

## Features

- **Real-time Voice Conversation**: Engage in natural back-and-forth conversation through your microphone
- **Local Execution**: All models run locally for privacy and offline use
- **Voice Cloning**: Uses Zonos TTS with voice embeddings to clone Daniel Dennett's distinctive voice
- **Large Language Model**: Powered by Meta's Llama 3.1 8B Instruct model for intelligent responses
- **Speech Recognition**: Uses OpenAI's Whisper model to transcribe your voice
- **Web-based Interface**: Clean, modern interface for easy interaction
- **Fallback Mechanisms**: Gracefully degrades to alternative TTS if models fail to load

## System Architecture

Paddington consists of three main components:

1. **Speech Recognition (Whisper)**: Transcribes user's voice input to text
2. **Language Model (Llama)**: Processes text and generates contextually relevant responses
3. **Text-to-Speech (Zonos)**: Converts text responses to spoken audio in Daniel Dennett's voice

These components are integrated through a FastAPI server with WebSocket support for real-time communication.

## Prerequisites

- Python 3.8+ (3.10 or 3.11 recommended)
- CUDA-capable GPU (recommended but not required)
- 8GB+ RAM
- 10GB+ disk space for models

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MurtazaKafka/paddington.git
   cd paddington
   ```

2. **Run the setup script**:
   ```bash
   python run_paddington.py
   ```
   This will install dependencies, set up directories, and start the server.

3. **Access the web interface** at `http://localhost:8000`

## Models Setup

Paddington requires three AI models:

### 1. Llama 3.1 Model
Download from [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main) and place at `/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf` or update the path in `.env`.

### 2. Zonos TTS Model
Download from [Hugging Face](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) and place at `/data/models/zonos_hybrid/` or update the path in `.env`.

### 3. Whisper Model
Automatically downloaded on first use.

## Manual Installation

For more control over the installation process:

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** by creating a `.env` file:
   ```
   ZONOS_MODEL_PATH=/path/to/zonos/model.safetensors
   ZONOS_CONFIG_PATH=/path/to/zonos/config.json
   LLM_MODEL_PATH=/path/to/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
   PERSONALITY_PATH=data/personality.json
   USE_SSL=False
   ```

4. **Start the server**:
   ```bash
   cd src
   python api.py
   ```

## Voice Customization

The default voice is configured to sound like philosopher Daniel Dennett. You can adjust voice parameters in `src/zonos_tts.py`:

```python
self.dennett_speaking_rate = 0.9    # Slower for a more methodical pace
self.dennett_pitch_std = 0.22       # Medium-low pitch variation
self.dennett_emotion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Neutral scholarly tone
```

For a different reference voice, replace the audio file at `data/reference/dennett_reference.wav` with your preferred speaker.

## Configuration

### Personality Configuration

The AI personality can be customized by editing `data/personality.json`. This affects the LLM's responses, making them match the desired personality traits.

### Reference Audio

For optimal voice cloning results:
1. Use a high-quality audio sample (5+ seconds) of the target voice
2. Place it in `data/reference/` directory
3. Update the path in the Simulator initialization if needed

## API Endpoints

Paddington provides several API endpoints:

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /process-text` - Process text without audio
- `POST /upload-audio` - Process audio input
- `GET /audio/{filename}` - Retrieve generated audio
- `WebSocket /ws` - Real-time audio streaming

## Troubleshooting

### Common Issues

1. **Missing Models**
   - Check that all model paths in `.env` are correct
   - Ensure you've downloaded the required models

2. **TTS Issues**
   - If Zonos fails to load, the system falls back to Google TTS
   - Check console logs for specific error messages

3. **CUDA/GPU Errors**
   - Try setting `device="cpu"` in the ZonosTTS initialization
   - Ensure your CUDA drivers are up to date

4. **Audio Not Working**
   - Check your browser's microphone permissions
   - Inspect browser console for JavaScript errors

### Logs

Check these log files for detailed information:
- `src/paddington_api.log`
- `src/websocket_server.log`

## Project Structure

```
paddington/
├── src/                      # Main source code
│   ├── api.py                # FastAPI server, routes, and endpoints
│   ├── websocket_server.py   # Real-time WebSocket handling
│   ├── simulator_llama.py    # Core logic integrating LLM with TTS
│   ├── zonos_tts.py          # Text-to-speech functionality
│   ├── index.html            # Web interface
│   ├── static/               # Static assets
│   └── uploads/              # Temporary audio uploads
├── data/
│   ├── personality.json      # AI personality configuration
│   └── reference/            # Voice reference audio files
├── venv/                     # Virtual environment
├── start_paddington.sh       # Startup script
├── run_paddington.py         # Setup and initialization script
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

## Privacy Considerations

Paddington processes all data locally on your machine:
- No data is sent to external servers
- Voice recordings are processed and then deleted
- No API keys or external services are required

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Zonos TTS system for voice synthesis
- Meta's Llama models for text generation
- OpenAI's Whisper model for speech recognition 