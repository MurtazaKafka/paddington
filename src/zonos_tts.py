import os
import torch
import torchaudio
import logging
import numpy as np
from typing import Optional
import traceback
import sys
import warnings
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_import_zonos():
    """Safely import Zonos by temporarily setting numpy version check"""
    try:
        # Save original numpy version
        original_numpy_version = np.__version__
        logger.info(f"Current NumPy version: {original_numpy_version}")
        
        # Make NumPy think it's an older version for Zonos import (which uses Numba)
        # Numba needs NumPy 1.26 or less
        if original_numpy_version.startswith("2."):
            logger.info("Setting NumPy version to 1.26.0 for Zonos compatibility")
            np.__version__ = "1.26.0"
        
        # Import Zonos modules
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        
        # Restore original numpy version
        np.__version__ = original_numpy_version
        
        return Zonos, make_cond_dict
    except Exception as e:
        logger.error(f"Failed to import Zonos: {e}")
        logger.error(traceback.format_exc())
        return None, None

class ZonosTTS:
    """ZonosTTS class for text-to-speech conversion using Zonos models with voice cloning"""
    
    def __init__(self, model_name: str = "zonos", device: Optional[str] = None,
                 reference_audio_path: Optional[str] = None):
        """
        Initialize the TTS system
        
        Args:
            model_name: Model name to use, defaults to "zonos"
            device: Device to use for processing, defaults to CUDA if available else CPU
            reference_audio_path: Path to reference audio for voice cloning
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set Daniel Dennett voice characteristics - more natural approach
        self.dennett_speaking_rate = 1.0    # Normal speaking rate
        self.dennett_pitch_std = 0.15       # Natural pitch variation
        self.dennett_emotion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Neutral scholarly tone
        self.dennett_fmax = 22050           # High quality audio
        
        # Initialize model and processor
        self.model = None
        self.speaker = None
        
        # Try to load Zonos
        logger.info(f"Loading TTS model: {model_name}")
        
        # Set up reference audio path for voice cloning
        self.reference_audio_path = reference_audio_path
        
        try:
            if model_name == "zonos":
                # Use safe import function to handle numpy version conflicts
                Zonos, make_cond_dict = safe_import_zonos()
                
                if Zonos is not None and make_cond_dict is not None:
                    logger.info("Successfully imported Zonos")
                    
                    # Load the Zonos model
                    logger.info("Loading Zonos model...")
                    self.model = Zonos.get_model(device=self.device)
                    self.make_cond_dict = make_cond_dict
                    
                    # Load reference audio for voice cloning if provided
                    if self.reference_audio_path and os.path.exists(self.reference_audio_path):
                        logger.info(f"Loading reference audio from {self.reference_audio_path}")
                        try:
                            # Process the reference audio for better voice cloning
                            reference_tensor = self.process_reference_audio()
                            if reference_tensor is not None:
                                # Extract speaker embedding
                                logger.info("Extracting speaker embedding from reference audio...")
                                self.speaker = self.model.get_speaker_embedding(reference_tensor)
                                logger.info("Speaker embedding extracted successfully")
                            else:
                                logger.warning("Could not process reference audio, using default voice")
                                self.speaker = None
                        except Exception as e:
                            logger.error(f"Failed to load reference audio: {str(e)}")
                            logger.error(traceback.format_exc())
                            # Default speaker (will use the model's default voice)
                            self.speaker = None
                    else:
                        logger.warning("No reference audio provided or file not found, using default voice")
                        self.speaker = None
                    
                    logger.info("Zonos TTS system initialized successfully")
                else:
                    logger.warning("Failed to import Zonos, falling back to transformers")
                    self._load_transformers_tts()
            else:
                # Use transformers for other model types
                logger.info(f"Using transformers for model: {model_name}")
                self._load_transformers_tts(model_name)
        except Exception as e:
            logger.error(f"Error initializing TTS system: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("Falling back to transformers TTS")
            self._load_transformers_tts()
    
    def generate_speech(self, text: str, output_path: str, voice_style: str = "dennett") -> Optional[str]:
        """
        Generate speech from text and save to output_path
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the generated audio
            voice_style: Voice style to use, defaults to "dennett"
            
        Returns:
            Path to the generated audio file or None if generation failed
        """
        logger.info(f"Generating speech for text: '{text}'")
        
        try:
            if self.model is None:
                logger.error("No TTS model loaded, cannot generate speech")
                return None
            
            # Create the output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            # Generate speech using Zonos
            try:
                logger.info("Generating speech with Zonos...")
                
                # Re-import dynamically to ensure we have the proper numpy version handling
                Zonos, make_cond_dict = safe_import_zonos()
                if Zonos is None or make_cond_dict is None:
                    raise ImportError("Failed to import Zonos modules")
                
                # Configure Daniel Dennett's voice characteristics
                if voice_style == "dennett":
                    cond_dict = make_cond_dict(
                        text=text,
                        speaker=self.speaker,  # Use extracted speaker embedding
                        speaking_rate=self.dennett_speaking_rate,
                        pitch_std=self.dennett_pitch_std,
                        fmax=self.dennett_fmax,
                        language="en-us",
                        emotion=self.dennett_emotion
                    )
                    logger.info(f"Using Daniel Dennett voice style with speaking rate {self.dennett_speaking_rate}, pitch {self.dennett_pitch_std}")
                else:
                    # Default settings for other voices
                    cond_dict = make_cond_dict(
                        text=text,
                        speaker=self.speaker,
                        language="en-us"
                    )
                
                # Generate audio
                with torch.no_grad():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Generate audio
                        audio_array = self.model.generate(**cond_dict)
                
                # Save audio to file
                # Convert audio to proper format (typically 16-bit PCM .wav)
                sampling_rate = 24000  # Zonos uses 24kHz
                
                # Save audio file
                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()
                
                # Ensure audio is within proper range for 16-bit audio
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9
                
                import soundfile as sf
                sf.write(output_path, audio_array, sampling_rate)
                
                logger.info(f"Generated speech saved to: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Error generating speech with Zonos: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Fall back to transformers if available
                if hasattr(self, 'tts_pipeline'):
                    logger.warning("Falling back to transformers TTS")
                    
                    # Generate using transformers
                    speech = self.tts_pipeline(text, forward_params={"do_sample": True})
                    speech_array = speech["audio"][0].numpy()
                    sample_rate = speech["sampling_rate"]
                    
                    import soundfile as sf
                    sf.write(output_path, speech_array, sample_rate)
                    
                    logger.info(f"Generated speech (using transformers fallback) saved to: {output_path}")
                    return output_path
                else:
                    logger.error("No fallback TTS available")
                    return None
                
        except Exception as e:
            logger.error(f"Failed to generate speech: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _load_transformers_tts(self, model_name: str = "microsoft/speecht5_tts"):
        """
        Helper method to load the transformers TTS model
        
        Args:
            model_name: The model name to load, defaults to "microsoft/speecht5_tts"
        """
        try:
            from transformers import pipeline, AutoProcessor, SpeechT5HifiGan
            
            logger.info(f"Loading transformers TTS model: {model_name}")
            
            # Explicitly set device_map to the CUDA device if available
            device_map = {"": 0} if self.device.type == "cuda" else "auto"
            
            # Initialize the text-to-speech pipeline
            self.tts_pipeline = pipeline(
                "text-to-speech", 
                model=model_name,
                device_map=device_map
            )
            
            # Get the vocoder
            logger.info("Loading vocoder...")
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                device_map=device_map
            )
            
            # Load the processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            logger.info("Transformers TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load transformers TTS: {str(e)}")
            logger.error(traceback.format_exc())
            self.tts_pipeline = None
            self.vocoder = None
            self.processor = None
            logger.warning("No TTS system available")

    def process_reference_audio(self):
        """Process reference audio for better voice cloning results"""
        if not self.reference_audio_path or not os.path.exists(self.reference_audio_path):
            logger.warning("No reference audio found or file doesn't exist")
            return
            
        try:
            import librosa
            logger.info(f"Processing reference audio from {self.reference_audio_path}")
            
            # Load reference audio
            reference_audio, sr = librosa.load(self.reference_audio_path, sr=None)
            
            # Ensure minimum duration (5+ seconds for voice cloning)
            if len(reference_audio) / sr < 5:
                logger.warning(f"Reference audio is short ({len(reference_audio)/sr:.2f}s), voice cloning may be less accurate")
                repeats = max(1, int(5 / (len(reference_audio) / sr)))
                reference_audio = np.tile(reference_audio, repeats)
                logger.info(f"Extended reference audio to {len(reference_audio)/sr:.2f}s by repeating {repeats} times")
                
            # Resample if needed
            if sr != 24000:  # Zonos expects 24kHz audio
                logger.info(f"Resampling reference audio from {sr}Hz to 24000Hz")
                reference_audio = librosa.resample(reference_audio, orig_sr=sr, target_sr=24000)
                sr = 24000
            
            # Normalize audio
            reference_audio = librosa.util.normalize(reference_audio)
            
            # Convert to torch tensor
            reference_tensor = torch.tensor(reference_audio).unsqueeze(0).to(self.device)
            logger.info(f"Reference audio processed: duration={len(reference_audio)/sr:.2f}s, shape={reference_tensor.shape}")
            
            # Save processed audio for debugging
            processed_path = self.reference_audio_path.replace('.wav', '_processed.wav')
            try:
                import soundfile as sf
                sf.write(processed_path, reference_audio, sr)
                logger.info(f"Saved processed reference audio to {processed_path}")
            except Exception as e:
                logger.warning(f"Could not save processed reference audio: {e}")
            
            return reference_tensor
            
        except Exception as e:
            logger.error(f"Error processing reference audio: {e}")
            logger.error(traceback.format_exc())
            return None 