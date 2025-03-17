from fastapi import WebSocket
import asyncio
import logging
from typing import Dict, Optional
import json
import os
from model_integration import PersonalitySimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioStreamManager:
    def __init__(self):
        """
        Initialize the audio stream manager
        """
        logger.info("Initializing AudioStreamManager...")
        self.active_connections: Dict[str, WebSocket] = {}
        logger.info("Loading PersonalitySimulator...")
        self.simulator = PersonalitySimulator(
            llama_model_path="/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
            voice_model_path="/data/models/voice/voice_model.pth",
            personality_data_path="/home/murtaza/paddington/data/personality.json"
        )
        logger.info("AudioStreamManager initialized successfully")

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Handle new WebSocket connections
        """
        logger.info(f"New connection attempt from client {client_id}")
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            logger.info(f"Client {client_id} connected successfully")
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            raise

    async def disconnect(self, client_id: str):
        """
        Handle WebSocket disconnections
        """
        logger.info(f"Disconnecting client {client_id}")
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].close()
                del self.active_connections[client_id]
                logger.info(f"Client {client_id} disconnected successfully")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

    async def process_audio(self, client_id: str, audio_data: bytes):
        """
        Process incoming audio and generate response
        """
        logger.info(f"Processing audio from client {client_id}")
        try:
            # Save incoming audio to temporary file
            temp_audio_path = os.path.join(self.simulator.audio_dir, f"temp_{client_id}.wav")
            logger.info(f"Saving temporary audio file to {temp_audio_path}")
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
            
            # First transcribe the audio
            logger.info("Transcribing audio...")
            transcribed_text = self.simulator.transcribe_audio(temp_audio_path)
            logger.info(f"Transcribed text: {transcribed_text}")
            
            # Add user message to conversation
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json({
                    "type": "user_message",
                    "text": transcribed_text
                })
            
            # Process the interaction
            logger.info("Generating response...")
            response = self.simulator.process_interaction(audio_input=False, input_text=transcribed_text)
            logger.info(f"Generated response: {response}")
            
            # Send response back to client
            if client_id in self.active_connections:
                logger.info("Sending response to client")
                await self.active_connections[client_id].send_json({
                    "type": "response",
                    "text": response["response_text"],
                    "audio_path": response["audio_path"]
                })
            
            # Clean up temporary file
            try:
                os.remove(temp_audio_path)
                logger.info("Cleaned up temporary audio file")
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json({
                    "type": "error",
                    "message": str(e)
                })

# Create a singleton instance
stream_manager = AudioStreamManager() 