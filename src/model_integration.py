import os
import torch
import whisper
from llama_cpp import Llama
import logging
from typing import Optional, Dict, Any
import json
import wave
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalitySimulator:
    def __init__(self, 
                 llama_model_path: str,
                 voice_model_path: str,
                 personality_data_path: Optional[str] = None):
        """
        Initialize the personality simulator with necessary models
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Create audio directory if it doesn't exist
        self.audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
        os.makedirs(self.audio_dir, exist_ok=True)
        logger.info(f"Audio files will be saved in: {self.audio_dir}")
        
        # Initialize LLaMA model
        self.llm = Llama(
            model_path=llama_model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=32
        )
        
        # Initialize Whisper for speech recognition
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
        
        # Load personality data if provided
        self.personality_data = None
        if personality_data_path:
            with open(personality_data_path, 'r') as f:
                self.personality_data = json.load(f)
                logger.info("Personality data loaded successfully")
        
        # Initialize voice model path
        self.voice_model_path = voice_model_path
        logger.info(f"Voice model path set to: {voice_model_path}")

    def create_dummy_audio(self, duration: float = 1.0, sample_rate: int = 16000) -> str:
        """
        Create a dummy audio file for testing purposes
        """
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        data = (data * 32767).astype(np.int16)  # Convert to 16-bit PCM
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_audio_{timestamp}.wav"
        output_path = os.path.join(self.audio_dir, filename)
        
        # Save as WAV file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(data.tobytes())
        
        logger.info(f"Created dummy audio file: {output_path}")
        return filename  # Return only the filename, not the full path

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio input using Whisper
        """
        try:
            # First check if ffmpeg is installed
            import shutil
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                logger.error("ffmpeg is not installed. It is required for audio transcription.")
                # In a real system, we would fail here, but for demonstration
                # let's return a placeholder message
                return "This is a placeholder transcription since ffmpeg is not installed. Please install ffmpeg to enable actual transcription."
            
            full_path = os.path.join(self.audio_dir, audio_path) if not os.path.isabs(audio_path) else audio_path
            result = self.whisper_model.transcribe(full_path)
            logger.info(f"Successfully transcribed audio: {result['text']}")
            return result["text"]
        except FileNotFoundError as e:
            if 'ffmpeg' in str(e):
                logger.error("ffmpeg is not installed. It is required for audio transcription.")
                return "This is a placeholder transcription since ffmpeg is not installed. Please install ffmpeg to enable actual transcription."
            else:
                logger.error(f"File not found error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def record_audio(self, duration: int = 5) -> str:
        """
        Simulate audio recording by creating a dummy audio file
        """
        try:
            logger.info(f"Creating dummy audio recording for {duration} seconds...")
            filename = self.create_dummy_audio(duration=float(duration))
            return filename
        except Exception as e:
            logger.error(f"Error creating dummy audio: {e}")
            raise

    def synthesize_speech(self, text: str, output_filename: str = None) -> str:
        """
        Create a dummy audio file for the response
        """
        try:
            logger.info(f"Creating dummy audio for text: {text}")
            
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"response_{timestamp}.wav"
            
            # Create a 2-second dummy audio file
            return self.create_dummy_audio(duration=2.0, sample_rate=16000)
            
        except Exception as e:
            logger.error(f"Error creating dummy audio: {e}")
            raise

    async def process_interaction(self, input_text: Optional[str] = None, audio_input: bool = False, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete interaction cycle
        """
        try:
            # Handle input
            if audio_input and audio_path:
                # Directly transcribe the provided audio file
                logger.info(f"Transcribing provided audio file: {audio_path}")
                input_text = self.transcribe_audio(audio_path)
            elif audio_input and not audio_path:
                # Record audio if no path provided (simulation mode)
                logger.info("No audio path provided, simulating recording")
                audio_filename = self.record_audio()
                input_text = self.transcribe_audio(audio_filename)
                # Clean up the temporary audio file
                try:
                    os.remove(os.path.join(self.audio_dir, audio_filename))
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file: {e}")
            elif input_text is None:
                raise ValueError("Either audio_input must be True with a valid audio_path, or input_text must be provided")
            
            logger.info(f"Processing input text: {input_text}")
            
            # Generate response
            response_text = self.generate_response(input_text)
            
            # Synthesize speech for the response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"response_{timestamp}.wav"
            audio_filename = self.synthesize_speech(response_text, audio_filename)
            
            # Construct full URL for audio (this will be used by the client)
            audio_url = f"/get-response-audio/{audio_filename}"
            
            return {
                "input_text": input_text,
                "text": response_text,  # Changed from response_text to text for consistency
                "audio_url": audio_url
            }
        except Exception as e:
            logger.error(f"Error in interaction: {e}", exc_info=True)
            raise

    def generate_response(self, input_text: str) -> str:
        """
        Generate text response using LLaMA in Daniel Dennett's style
        """
        try:
            # Create a comprehensive prompt with personality context
            if self.personality_data:
                logger.info("Using personality data for response generation")
                context = f"""You are Daniel Dennett, speaking in the present moment. Background: {self.personality_data['background']}

Your philosophical positions:
- {self.personality_data['philosophical_positions']['consciousness']}
- {self.personality_data['philosophical_positions']['free_will']}
- {self.personality_data['philosophical_positions']['evolution']}
- {self.personality_data['philosophical_positions']['mind']}

Your speaking style is {self.personality_data['speech_patterns']['speaking_style']}

When responding, use your characteristic approach:
1. Break down complex ideas systematically
2. Use thought experiments and analogies when helpful
3. Maintain your materialist perspective
4. Draw from your expertise in {', '.join(self.personality_data['knowledge_domains'])}
5. Stay true to your philosophical positions while being engaging

Remember to occasionally use phrases like: {', '.join(self.personality_data['speech_patterns']['common_phrases'])}

Now, please respond to this query: {input_text}

Response:"""
            else:
                logger.warning("No personality data available, using basic prompt")
                context = ""
            
            prompt = f"{context}User: {input_text}\nDaniel Dennett:"
            logger.info(f"Generated prompt: {prompt}")
            
            logger.info("Calling LLaMA model for response generation...")
            response = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.8,
                stop=["User:", "\n\n\n"]
            )
            logger.info(f"Raw LLaMA response: {response}")
            
            result = response['choices'][0]['text'].strip()
            logger.info(f"Processed response: {result}")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response."

if __name__ == "__main__":
    # Example usage
    simulator = PersonalitySimulator(
        llama_model_path="/path/to/llama/model",
        voice_model_path="/path/to/voice/model",
        personality_data_path="/path/to/personality/data.json"
    )
    
    result = simulator.process_interaction(audio_input=True)
    print(result) 