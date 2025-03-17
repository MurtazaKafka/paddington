# Paddington - AI Voice Chat Assistant

Paddington is an AI-powered voice chat assistant that combines state-of-the-art language models with speech recognition and synthesis capabilities. It enables real-time voice conversations with an AI that has a distinct personality and can engage in natural, context-aware dialogue.

## Features

- Real-time voice chat using WebSocket connections
- Speech-to-text using OpenAI's Whisper model
- Text generation using Meta's Llama 3.1 8B Instruct model
- Text-to-speech synthesis
- Web-based user interface for easy interaction
- Customizable AI personality through JSON configuration

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Node.js and npm (for frontend development)

## Required Models

The following models need to be placed in the specified directories:

- LLaMA Model: `/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf`
- Voice Model: `/data/models/voice/voice_model.pth`
- Personality Data: `/home/murtaza/paddington/data/personality.json`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MurtazaKafka/paddington.git
cd paddington
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
paddington/
├── src/
│   ├── api.py              # FastAPI server and endpoints
│   ├── model_integration.py # AI model integration
│   ├── websocket_server.py # WebSocket handling
│   └── static/
│       └── index.html      # Web interface
├── data/
│   └── personality.json    # AI personality configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

1. Personality Configuration:
   - Edit `data/personality.json` to customize the AI's personality traits, conversation style, and behavior.

2. Model Paths:
   - Update the model paths in `src/api.py` if your models are stored in different locations.

## Running the Application

1. Start the server:
```bash
cd src
python api.py
```

2. Access the web interface:
   - Open a web browser and navigate to `http://localhost:8000`
   - Click the microphone button to start recording
   - Speak your message
   - Click the button again to stop recording and send the audio
   - Wait for the AI's response

## API Endpoints

- `GET /` - Serves the web interface
- `GET /health` - Health check endpoint
- `POST /process-text` - Process text input
- `POST /process-audio` - Process audio input
- `GET /get-response-audio/{filename}` - Retrieve generated audio responses
- `WebSocket /ws` - Real-time audio streaming endpoint

## Development

### Frontend Development
The frontend is built using vanilla JavaScript and HTML5 Web APIs:
- WebSocket for real-time communication
- MediaRecorder for audio capture
- Web Audio API for audio playback

### Backend Development
The backend is built with:
- FastAPI for the web server
- Whisper for speech recognition
- Llama for text generation
- Custom WebSocket handler for real-time communication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI's Whisper model for speech recognition
- Meta's Llama model for text generation
- FastAPI framework for the backend server
- The open-source community for various tools and libraries used in this project 