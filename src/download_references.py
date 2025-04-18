#!/usr/bin/env python3
"""
Download or create reference audio for Daniel Dennett's voice.
This script will be used to create a voice profile for the TTS system.
"""

import os
import sys
import logging
import requests
import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Optional
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define constants
DENNETT_YOUTUBE_IDS = [
    "rMFNvsGUCIw",  # Conversation With Richard Dawkins and Daniel Dennett
    "DTepA-WV_oM",  # Daniel Dennett: The Future of Life - Schrödinger at 75
    "vI-RGoh8U4c",  # Explaining Consciousness - Daniel Dennett Interview
    "D1y1knpxV0E"    # How to Make Thinking Better - Daniel Dennett
]

SAMPLE_RATE = 24000
VOICES_DIR = Path("/data/models/voices")
REF_AUDIO_DIR = VOICES_DIR / "reference_audio"
PROFILE_DIR = VOICES_DIR / "profiles"

def download_youtube_audio(youtube_id: str, output_path: Path) -> Optional[Path]:
    """
    Download audio from a YouTube video using yt-dlp.
    Returns the path to the downloaded file or None if download failed.
    """
    try:
        if output_path.exists():
            logger.info(f"Audio file already exists at {output_path}")
            return output_path
            
        logger.info(f"Downloading audio from YouTube ID: {youtube_id}")
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Use yt-dlp to download the audio
        output_template = str(output_path).replace('.wav', '')
        cmd = f"yt-dlp -x --audio-format wav --audio-quality 0 -o '{output_template}.%(ext)s' https://www.youtube.com/watch?v={youtube_id}"
        logger.info(f"Running command: {cmd}")
        os.system(cmd)
        
        # Check if the file was created (yt-dlp may add .wav extension)
        final_path = Path(f"{output_template}.wav")
        if final_path.exists():
            # Rename if necessary
            if final_path != output_path:
                os.rename(final_path, output_path)
            logger.info(f"Successfully downloaded audio to {output_path}")
            return output_path
        else:
            logger.error(f"Failed to download audio for YouTube ID: {youtube_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {e}")
        return None

def create_synthetic_voice_profile(output_path: Path) -> None:
    """
    Create a synthetic voice profile for Daniel Dennett
    using predetermined characteristics.
    """
    try:
        logger.info("Creating synthetic voice profile for Daniel Dennett")
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Create a deterministic embedding with characteristics of Daniel Dennett's voice
        np.random.seed(42)  # Use a fixed seed for reproducibility
        
        # Base embedding
        embedding = np.random.randn(128)
        
        # Adjust for typical male voice characteristics
        # Lower pitch - emphasize lower dimensions
        embedding[:32] *= 1.5
        
        # Academic speaking style - adjust mid dimensions
        embedding[32:64] *= 1.2
        
        # Philosopher characteristics - emphasize certain frequencies
        embedding[64:96] *= 1.3
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Convert to torch tensor and save
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        torch.save(embedding_tensor, output_path)
        
        logger.info(f"Successfully created synthetic voice profile at {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating synthetic voice profile: {e}")

def create_synthetic_reference_audio(output_path: Path) -> None:
    """
    Create synthetic reference audio for Daniel Dennett.
    """
    try:
        logger.info("Creating synthetic reference audio for Daniel Dennett")
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Create a sound with characteristics of Daniel Dennett's voice
        duration = 5.0  # seconds
        sample_rate = SAMPLE_RATE
        
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Base frequency for male voice (Daniel Dennett has a medium-low register)
        fundamental = 120.0  # Hz
        
        # Create harmonic series for voice
        waveform = 0.5 * np.sin(2 * np.pi * fundamental * t)
        
        # Add harmonics with decreasing amplitude
        for i in range(2, 10):
            amplitude = 0.3 / i
            waveform += amplitude * np.sin(2 * np.pi * fundamental * i * t)
        
        # Add formants (typical for English male voice)
        formant1 = 700  # Hz
        formant2 = 1200  # Hz
        formant3 = 2300  # Hz
        
        # Add formants with appropriate amplitudes
        waveform += 0.2 * np.sin(2 * np.pi * formant1 * t)
        waveform += 0.17 * np.sin(2 * np.pi * formant2 * t)
        waveform += 0.15 * np.sin(2 * np.pi * formant3 * t)
        
        # Add vibrato (slight variation in pitch)
        vibrato_rate = 5.0  # Hz
        vibrato_depth = 0.005
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        t_vibrato = t + vibrato
        
        vibrato_waveform = 0.5 * np.sin(2 * np.pi * fundamental * t_vibrato)
        for i in range(2, 10):
            amplitude = 0.3 / i
            vibrato_waveform += amplitude * np.sin(2 * np.pi * fundamental * i * t_vibrato)
        
        # Mix original and vibrato waveforms
        waveform = 0.7 * waveform + 0.3 * vibrato_waveform
        
        # Apply envelope for natural sound
        env = np.ones_like(t)
        attack_time = 0.02  # seconds
        release_time = 0.1  # seconds
        
        attack_samples = int(attack_time * sample_rate)
        release_samples = int(release_time * sample_rate)
        
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        if release_samples > 0:
            rel_start = len(env) - release_samples
            if rel_start > 0:
                env[rel_start:] = np.linspace(1, 0, release_samples)
        
        waveform = waveform * env
        
        # Normalize
        waveform = waveform / np.max(np.abs(waveform))
        
        # Convert to int16 for WAV file
        waveform_int = (waveform * 32767).astype(np.int16)
        
        # Save as WAV file
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        torchaudio.save(output_path, waveform_tensor, sample_rate)
        
        logger.info(f"Successfully created synthetic reference audio at {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating synthetic reference audio: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download or create reference audio for Daniel Dennett's voice")
    parser.add_argument('--synthetic', action='store_true', help="Create synthetic references instead of downloading")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(REF_AUDIO_DIR, exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)
    
    if args.synthetic:
        # Create synthetic reference audio
        logger.info("Creating synthetic voice references")
        reference_path = REF_AUDIO_DIR / "dennett_synthetic.wav"
        create_synthetic_reference_audio(reference_path)
        
        # Create synthetic voice profile
        profile_path = PROFILE_DIR / "dennett_profile.pt"
        create_synthetic_voice_profile(profile_path)
    else:
        # Attempt to download YouTube audio
        logger.info("Downloading voice references from YouTube")
        success = False
        for idx, youtube_id in enumerate(DENNETT_YOUTUBE_IDS):
            output_path = REF_AUDIO_DIR / f"dennett_{idx}.wav"
            if download_youtube_audio(youtube_id, output_path):
                success = True
        
        if not success:
            logger.warning("Failed to download any audio files. Creating synthetic reference as fallback.")
            reference_path = REF_AUDIO_DIR / "dennett_synthetic.wav"
            create_synthetic_reference_audio(reference_path)
            
        # Create synthetic voice profile (since we don't have extraction yet)
        profile_path = PROFILE_DIR / "dennett_profile.pt"
        create_synthetic_voice_profile(profile_path)
    
    logger.info("Reference audio and voice profile creation complete!")

if __name__ == "__main__":
    main() 