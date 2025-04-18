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
from pathlib import Path
import torchaudio

# Import llama-cpp-python for local LLM
from llama_cpp import Llama

# Force CPU usage to avoid CUDA memory issues - comment out to allow TTS to use CUDA
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# torch.set_default_device("cpu")

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, model_path: str, config_path: str):
        # Get the correct path for static files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "static", "audio")
        logger.info(f"Audio output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Use the Big Think reference audio of Daniel Dennett for voice cloning
        reference_audio_path = os.path.join(os.path.dirname(current_dir), "data", "reference", "dennett_reference.wav")
        logger.info(f"Reference audio path: {reference_audio_path}")
        
        try:
            # Initialize TTS with hybrid model for better voice cloning and cuda support
            logger.info("Loading Zonos-v0.1-hybrid model for better voice cloning...")
            self.tts = ZonosTTS(model_name="Zyphra/Zonos-v0.1-hybrid", device="cuda", reference_audio_path=reference_audio_path)
            
            # Set Daniel Dennett specific voice characteristics
            self.tts.dennett_speaking_rate = 0.9  # Slower methodical pace
            self.tts.dennett_pitch_std = 0.22     # Medium-low pitch variation for Dennett's distinctive voice
        except Exception as e:
            logger.error(f"Failed to load hybrid model: {e}. Falling back to transformer model.")
            # Fall back to transformer model if hybrid fails
            self.tts = ZonosTTS(model_name="Zyphra/Zonos-v0.1-transformer", device="cpu", reference_audio_path=reference_audio_path)
        
        logger.info("Loading Whisper model (this may take a moment)...")
        self.whisper_model = whisper.load_model("base", device="cpu")
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
        
        # Initialize Llama model - using absolute path where we found the model
        llama_model_path = "/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
        if os.path.exists(llama_model_path):
            logger.info(f"Initializing Llama model from {llama_model_path}")
            try:
                # Load the model with CPU only (n_gpu_layers=0)
                self.llm = Llama(
                    model_path=llama_model_path,
                    n_ctx=2048,
                    n_gpu_layers=0,  # No GPU layers, CPU only
                    verbose=False
                )
                logger.info("Llama model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Llama model: {str(e)}", exc_info=True)
                self.llm = None
        else:
            logger.error(f"Llama model not found at {llama_model_path}")
            self.llm = None
            
        # Set up Paddington system prompt
        self.system_prompt = """
        You are Paddington, an AI assistant who sounds just like philosopher Daniel Dennett.
        You should respond conversationally and naturally as if you were Daniel speaking.
        Keep your responses relatively concise (1-3 sentences when possible).
        Your philosophical positions include compatibilism about free will, 
        a materialist view of consciousness, and a computational theory of mind.
        """
        
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
        Generate response using Llama model
        
        Args:
            text: Input text
            
        Returns:
            Generated response text
        """
        try:
            if self.llm:
                # Format the prompt for Llama
                full_prompt = f"""
                <|system|>
                {self.system_prompt}
                </s>
                
                <|user|>
                {text}
                </s>
                
                <|assistant|>
                """
                
                # Log the prompt being sent
                logger.info(f"Sending prompt to Llama: {full_prompt}")
                
                # Generate response using Llama
                response = self.llm(
                    full_prompt,
                    max_tokens=200,
                    stop=["</s>", "<|user|>"],
                    temperature=0.7,
                    top_p=0.95,
                )
                
                # Extract the generated text
                generated_text = response["choices"][0]["text"].strip()
                logger.info(f"Llama Response: {generated_text}")
                
                # Update chat history
                self.chat_history.append({"role": "user", "content": text})
                self.chat_history.append({"role": "assistant", "content": generated_text})
                
                return generated_text
            else:
                # Return dummy response for testing
                logger.info("Using dummy response (Llama model not available)")
                dummy_response = "Consciousness isn't just something that happens in our heads. It's an emergent property of complex systems. I've been thinking about this for decades."
                
                # Update chat history with dummy content
                self.chat_history.append({"role": "user", "content": text})
                self.chat_history.append({"role": "assistant", "content": dummy_response})
                
                return dummy_response
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response."
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input and generate response
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing text response and audio URL
        """
        try:
            logger.info(f"Processing text input: {text}")
            
            # Generate LLM response
            response_text = await self.generate_llm_response(text)
            logger.info(f"Generated response: {response_text}")
            
            # Generate audio response using Zonos TTS
            audio_output_path = await self._generate_speech(response_text)
            logger.info(f"Speech generated successfully at: {audio_output_path}")
            
            # Return response with correct audio_url format
            filename = os.path.basename(audio_output_path)
            direct_audio_url = f"/audio/{filename}"
            
            # Construct response
            response = {
                "text": response_text,
                "audio_url": direct_audio_url
            }
            logger.info(f"Returning response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}", exc_info=True)
            # Return error response that frontend can handle
            return {
                "text": "I apologize, but I encountered an error while processing your request.",
                "error": str(e)
            }
            
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
            # Use direct path format consistently
            direct_audio_url = f"/audio/{filename}"
            
            logger.info(f"Audio filename: {filename}")
            logger.info(f"Direct audio URL: {direct_audio_url}")
            logger.info(f"Full server path: {os.path.abspath(audio_output_path)}")
            
            # Construct response
            response = {
                "text": response_text,
                "transcription": transcribed_text,
                "audio_url": direct_audio_url
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            # Return error response that frontend can handle
            return {
                "text": "I apologize, but I encountered an error while processing your audio.",
                "error": str(e)
            } 

    async def _generate_speech(self, text: str) -> str:
        """Generate speech from text using TTS"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            output_path = os.path.join(self.output_dir, f"{file_id}.wav")
            
            # Get current directory for absolute paths
            logger.info(f"Generating speech for text: {text}")
            logger.info(f"Output path: {output_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Pre-process text for better speech generation
            if not text.strip().endswith((".", "!", "?", ":", ";")):
                text = text.strip() + "."
            
            # Use TTS to generate speech
            # Run in a separate thread to avoid blocking
            try:
                await asyncio.to_thread(
                    self.tts.generate_speech,
                    text=text,
                    output_path=output_path,
                    voice_style="dennett"
                )
                
                # Verify the file exists and has content
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"Generated speech saved to {output_path} (size: {file_size} bytes)")
                    
                    if file_size == 0:
                        logger.error("Generated audio file is empty (0 bytes)")
                        raise ValueError("Empty audio file generated")
                else:
                    logger.error(f"Generated audio file does not exist: {output_path}")
                    raise FileNotFoundError(f"Audio file not created: {output_path}")
                
            except Exception as tts_error:
                logger.error(f"Error in TTS generation: {str(tts_error)}")
                # If there's an error, create a simple audio file
                from gtts import gTTS
                logger.info("Falling back to Google TTS...")
                
                # Change extension to mp3 for gTTS
                output_path = os.path.join(self.output_dir, f"{file_id}.mp3")
                
                # Create a simple TTS as fallback
                fallback_tts = gTTS(text=text, lang="en", slow=False)
                fallback_tts.save(output_path)
                logger.info(f"Created fallback audio with Google TTS: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in generate_speech: {str(e)}", exc_info=True)
            # Last resort fallback
            try:
                from gtts import gTTS
                fallback_path = os.path.join(self.output_dir, f"fallback_{uuid.uuid4()}.mp3")
                
                # Use even simpler text in case the problem was in the input text
                simple_text = "I'm sorry, I encountered an error generating speech."
                fallback_tts = gTTS(text=simple_text, lang="en", slow=False)
                fallback_tts.save(fallback_path)
                logger.info(f"Created emergency fallback audio: {fallback_path}")
                return fallback_path
            except Exception as final_error:
                logger.error(f"Final fallback also failed: {str(final_error)}")
                raise 