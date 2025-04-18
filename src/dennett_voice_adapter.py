"""
Daniel Dennett Voice Adapter

This module integrates the extracted Daniel Dennett voice profile
with the ZonosTTS system for improved voice synthesis.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class DennettVoiceAdapter:
    """Adapter class to apply Daniel Dennett's voice characteristics to TTS output"""
    
    def __init__(self):
        self.voice_profile_path = Path("data/tts/voices/dennett.json")
        self.voice_profile = self._load_voice_profile()
        
    def _load_voice_profile(self):
        """Load the voice profile from file"""
        try:
            if os.path.exists(self.voice_profile_path):
                with open(self.voice_profile_path, 'r') as f:
                    profile = json.load(f)
                logger.info(f"Loaded voice profile for {profile.get('name', 'unknown')}")
                return profile
            else:
                # Check alternate location
                alt_path = Path("data/voice_samples/voice_profile/dennett_voice_profile.json")
                if os.path.exists(alt_path):
                    with open(alt_path, 'r') as f:
                        profile = json.load(f)
                    logger.info(f"Loaded voice profile from alternate location")
                    return profile
                else:
                    logger.warning("Voice profile not found, using default settings")
                    return self._create_fallback_profile()
        except Exception as e:
            logger.error(f"Error loading voice profile: {e}")
            return self._create_fallback_profile()
    
    def _create_fallback_profile(self):
        """Create a fallback profile based on known characteristics of Dennett's voice"""
        # These values are approximations based on typical older male academic voice
        return {
            "voice_id": "daniel_dennett_fallback",
            "name": "Daniel Dennett (Fallback)",
            "description": "Fallback voice profile for philosopher Daniel Dennett",
            "features": {
                # Typical values for an older male academic
                "pitch_mean": 120.0,         # Lower pitch (Hz)
                "pitch_std": 15.0,           # Moderate pitch variation
                "pitch_min": 85.0,           # Lower end of pitch range
                "pitch_max": 180.0,          # Upper end of pitch range
                "spectral_centroid_mean": 950.0,  # Darker timbre
                "spectral_bandwidth_mean": 1500.0,  # Moderate bandwidth
                "tempo": 160.0,              # Deliberate speaking pace
                "energy_mean": 0.08,         # Moderate energy
                "energy_std": 0.04,          # Some dynamic range
                "rms_mean": 0.12,            # Moderate volume
                "rms_std": 0.05,             # Some volume variation
                "mfccs": [                   # Approximated MFCC values
                    -5.0, 60.0, -5.0, 0.5, -10.0, -5.0, 2.0, 
                    -2.0, -1.0, 0.5, 0.5, 0.0, -0.5
                ]
            }
        }
    
    def get_voice_parameters(self):
        """Get voice parameters for TTS synthesis"""
        if not self.voice_profile or 'features' not in self.voice_profile:
            return self._get_fallback_parameters()
            
        features = self.voice_profile.get('features', {})
        
        # Convert voice profile features to TTS parameters
        # These mappings are specific to the ZonosTTS system
        params = {
            "pitch_shift": self._normalize_pitch(features.get('pitch_mean', 120.0)),
            "pitch_range": features.get('pitch_std', 15.0) / 10.0,  # Scale to appropriate range
            "speed": self._map_tempo_to_speed(features.get('tempo', 160.0)),
            "energy": features.get('energy_mean', 0.08) * 10,  # Scale to 0-1 range
            "breathiness": 0.05,  # Fixed value for Dennett's clear articulation
            "roughness": 0.2,     # Slight roughness typical of older voice
            "temperature": 0.6,   # Moderate variation for natural speech
            "top_p": 0.8          # Moderate focus for consistent style
        }
        
        return params
    
    def _normalize_pitch(self, pitch_mean):
        """Normalize the pitch value to a shift factor"""
        # Convert Hz to relative shift factor where 1.0 is neutral
        # Base on typical male voice of ~120Hz
        reference_pitch = 120.0
        return pitch_mean / reference_pitch
    
    def _map_tempo_to_speed(self, tempo):
        """Map tempo (BPM) to speech speed factor"""
        # Map typical tempo range (120-180 BPM) to speed factor (0.8-1.2)
        # where 1.0 is neutral speed
        if tempo < 120:
            return 0.8 + (tempo - 80) / 100
        elif tempo > 180:
            return 1.2 + (tempo - 180) / 100
        else:
            return 0.8 + (tempo - 120) / 150
    
    def _get_fallback_parameters(self):
        """Get fallback parameters if profile isn't available"""
        return {
            "pitch_shift": 0.95,   # Slightly lower pitch (Dennett has a lower voice)
            "pitch_range": 1.5,    # Decent variation for emphasis
            "speed": 0.92,         # Slightly slower, deliberate speech
            "energy": 0.7,         # Moderate energy
            "breathiness": 0.05,   # Clear articulation
            "roughness": 0.2,      # Slight roughness
            "temperature": 0.6,    # Moderate variation
            "top_p": 0.8           # Moderate focus
        }
    
    def dennett_style_parameters(self):
        """Get comprehensive parameters for Dennett's speaking style"""
        voice_params = self.get_voice_parameters()
        
        # Add linguistic style parameters specific to Dennett
        style_params = {
            # Voice parameters from voice profile
            **voice_params,
            
            # Linguistic style parameters (not derived from audio)
            "word_rate": 0.92,             # Slightly measured pace
            "emphasis_level": 1.1,         # Slightly more emphasis on key words
            "pause_frequency": 1.2,        # More frequent pauses for emphasis
            "pause_duration": 1.15,        # Slightly longer pauses
            "sentence_final_pitch": 0.92,  # Declining pitch at end of sentences
            "question_final_pitch": 1.15,  # Rising pitch for questions
            "articulation_rate": 0.96,     # Clear articulation
            "cadence": "academic"          # Academic speaking style
        }
        
        return style_params
    
    def apply_voice_to_tts_config(self, tts_config):
        """Apply Dennett's voice parameters to a TTS configuration"""
        style_params = self.dennett_style_parameters()
        
        # Update the TTS configuration with Dennett's voice parameters
        if 'voice' not in tts_config:
            tts_config['voice'] = {}
            
        # Update voice parameters
        tts_config['voice'].update({
            "id": "daniel_dennett",
            "name": "Daniel Dennett",
            "pitch_shift": style_params["pitch_shift"],
            "speed": style_params["speed"],
            "energy": style_params["energy"],
            "pitch_range": style_params["pitch_range"],
            "temperature": style_params["temperature"],
            "top_p": style_params["top_p"]
        })
        
        return tts_config 
        
    def get_reference_audio_path(self):
        """
        Get the path to a reference audio file for Dennett's voice
        
        Returns:
            str: Path to the reference audio file or None if not found
        """
        # Check several possible locations for reference audio
        possible_paths = [
            "data/voice_samples/dennett_sample.wav",
            "data/voice_samples/dennett_sample.mp3",
            "data/tts/voice_samples/dennett.wav",
            "data/tts/voice_samples/dennett.mp3",
            "static/audio/dennett_sample.wav",
            "static/audio/dennett_sample.mp3"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found reference audio at: {path}")
                return path
                
        logger.warning("No reference audio file found for Dennett's voice")
        return None 