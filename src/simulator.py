import os
import logging
from typing import Optional, Dict, Any
from zonos_tts import ZonosTTS
import torch
import whisper
from datetime import datetime
import json
import uuid
import asyncio
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, model_path: str, config_path: str):
        # Get the correct path for static files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "static", "audio")
        logger.info(f"Audio output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize TTS
        self.tts = ZonosTTS(model_path, config_path)
        
        # Initialize Whisper model for transcription
        logger.info("Loading Whisper model (this may take a moment)...")
        self.whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully!")
        
        # Get voice parameters from TTS model
        # If the TTS model has loaded specific voice characteristics, use those
        if hasattr(self.tts, "dennett_characteristics"):
            logger.info("Using voice characteristics from profile")
            self.dennett_style = {
                "speaking_rate": getattr(self.tts, "dennett_speaking_rate", 0.9),
                "pitch_std": getattr(self.tts, "dennett_pitch_std", 1.2),
                "fmax": 20000.0,
                "emotion": torch.zeros(8)  # Neutral emotion
            }
            logger.info(f"Voice characteristics: {self.dennett_style}")
        else:
            # Default voice characteristics if not available from TTS
            logger.info("Using default voice characteristics")
            self.dennett_style = {
                "speaking_rate": 0.9,  # Slightly slower, methodical speaking
                "pitch_std": 1.2,  # Moderate pitch variation
                "fmax": 20000.0,  # Natural frequency range
                "emotion": torch.zeros(8)  # Neutral emotion
            }
        
        self.chat_history = []
        
        # Initialize OpenAI client
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        if OPENAI_API_KEY:
            logger.info("Initializing OpenAI client")
            self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            self.llm = ChatOpenAI(api_key=OPENAI_API_KEY)
        else:
            logger.warning("OPENAI_API_KEY is not set. Using dummy responses.")
            self.client = None
            self.llm = None
            
        # Set up Paddington system prompt
        self.system_prompt = """
        You are Paddington, an AI assistant who sounds just like philosopher Daniel Dennett.
        You should respond conversationally and naturally as if you were Daniel speaking.
        Keep your responses relatively concise (1-3 sentences when possible).
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper model
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio from: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            transcribed_text = result["text"].strip()
            logger.info(f"Transcription result: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            # Return a default message if transcription fails
            return "I couldn't understand what you said, but I'll respond anyway."
            
    async def generate_llm_response(self, text: str) -> str:
        """
        Generate response using LLM
        
        Args:
            text: Input text
            
        Returns:
            Generated response text
        """
        try:
            if self.llm:
                # Use OpenAI for response
                chain = self.prompt | self.llm
                response = chain.invoke({"input": text, "chat_history": self.chat_history})
                
                # Update chat history
                self.chat_history.append(HumanMessage(content=text))
                self.chat_history.append(AIMessage(content=response.content))
                
                logger.info(f"LLM Response: {response.content}")
                return response.content
            else:
                # Return dummy response for testing
                logger.info("Using dummy response (OpenAI API key not set)")
                dummy_response = "Consciousness isn't just something that happens in our heads. It's an emergent property of complex systems. I've been thinking about this for decades."
                
                # Update chat history with dummy content
                self.chat_history.append(HumanMessage(content=text))
                self.chat_history.append(AIMessage(content=dummy_response))
                
                return dummy_response
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response."
            
    async def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Process audio input and generate response
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Dictionary containing text response and audio URL
        """
        try:
            logger.info(f"Processing audio from: {audio_path}")
            
            # Transcribe audio to text
            transcribed_text = await self.transcribe_audio(audio_path)
            logger.info(f"Transcribed text: {transcribed_text}")
            
            # Generate LLM response
            response_text = await self.generate_llm_response(transcribed_text)
            logger.info(f"Generated response: {response_text}")
            
            # Generate audio response using Zonos TTS
            audio_output_path = await self._generate_speech(response_text)
            logger.info(f"Speech generated successfully at: {audio_output_path}")
            
            # Verify file exists
            if not os.path.exists(audio_output_path):
                logger.error(f"Generated audio file does not exist at: {audio_output_path}")
            
            # Check file size
            if os.path.exists(audio_output_path):
                file_size = os.path.getsize(audio_output_path)
                logger.info(f"Generated audio file size: {file_size} bytes")
            
            # Return response with correct audio_url format
            filename = os.path.basename(audio_output_path)
            static_audio_url = f"/static/audio/{filename}"
            direct_audio_url = f"/audio/{filename}"
            
            logger.info(f"Audio filename: {filename}")
            logger.info(f"Static audio URL: {static_audio_url}")
            logger.info(f"Direct audio URL: {direct_audio_url}")
            logger.info(f"Full server path: {os.path.abspath(audio_output_path)}")
            
            # Make sure the file exists and is readable
            if os.path.exists(audio_output_path) and os.access(audio_output_path, os.R_OK):
                logger.info(f"Audio file exists and is readable")
            else:
                logger.error(f"Audio file exists: {os.path.exists(audio_output_path)}, readable: {os.access(audio_output_path, os.R_OK) if os.path.exists(audio_output_path) else False}")
            
            # Construct response - use the direct URL format that works better with browsers
            response = {
                "text": response_text,
                "transcription": transcribed_text,
                "audio_url": direct_audio_url
            }
            logger.info(f"Returning response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}", exc_info=True)
            return {
                "text": "I apologize, but I encountered an error while processing your audio.",
                "error": str(e)
            }
    
    async def _generate_speech(self, text: str) -> str:
        """
        Generate speech from text using the TTS model

        Args:
            text: Text to convert to speech
            
        Returns:
            Path to the generated audio file
        """
        try:
            logger.info(f"Generating speech for text: {text}")
            
            # Create a unique filename for this audio response
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"response_{timestamp}_{unique_id}.wav"
            
            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, filename)
            
            logger.info(f"Will save audio to: {output_path}")
            
            # Apply Dennett voice characteristics if available
            if hasattr(self, 'dennett_style'):
                logger.info("Applying voice characteristics for Daniel Dennett style")
                
                # Generate the audio with the specific voice characteristics
                audio_array = self.tts.generate_speech(
                    text=text,
                    output_path=output_path,
                    language="en-us"
                )
                
                logger.info(f"Speech generation complete, saved to {output_path}")
                return output_path
            else:
                # Fallback to default voice settings
                logger.warning("No voice characteristics available, using defaults")
                audio_array = self.tts.generate_speech(
                    text=text,
                    output_path=output_path,
                    language="en-us"
                )
                logger.info(f"Speech generation complete with default voice, saved to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            
            # Create a simple fallback audio file with a message
            try:
                # Try to generate a simple error message audio or use a pre-recorded fallback
                fallback_path = os.path.join(self.output_dir, "error_fallback.wav")
                
                # If we already have a fallback file, use it
                if os.path.exists(fallback_path):
                    logger.info(f"Using existing fallback audio: {fallback_path}")
                    return fallback_path
                    
                # Otherwise, try to generate a simple error message
                if self.tts:
                    error_message = "I'm sorry, I encountered an error generating audio."
                    self.tts.generate_speech(
                        text=error_message,
                        output_path=fallback_path,
                        language="en-us"
                    )
                    logger.info(f"Generated fallback audio: {fallback_path}")
                    return fallback_path
            except Exception as fallback_error:
                logger.error(f"Failed to generate fallback audio: {str(fallback_error)}")
            
            # Last resort - return a path that will be handled by the error checking in process_audio
            return os.path.join(self.output_dir, "nonexistent_error_audio.wav") 
