#!/usr/bin/env python3
"""
Process Daniel Dennett voice samples to create a more realistic voice profile.
This script analyzes audio samples and extracts voice characteristics.
"""

import os
import sys
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import librosa
import soundfile as sf
from typing import List, Dict, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define constants
VOICES_DIR = Path("/data/models/voices")
SAMPLES_DIR = Path("data/voice_samples")
REF_AUDIO_DIR = VOICES_DIR / "reference_audio"
PROFILE_DIR = VOICES_DIR / "profiles"
SAMPLE_RATE = 24000

def prepare_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(REF_AUDIO_DIR, exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)
    logger.info(f"Directories created/verified: {SAMPLES_DIR}, {REF_AUDIO_DIR}, {PROFILE_DIR}")

def find_voice_samples() -> List[Path]:
    """Find all voice samples in the samples directory"""
    extensions = ['.wav', '.mp3', '.ogg', '.flac']
    samples = []
    
    for ext in extensions:
        samples.extend(list(SAMPLES_DIR.glob(f"*{ext}")))
    
    logger.info(f"Found {len(samples)} voice samples: {[s.name for s in samples]}")
    return samples

def process_audio_sample(sample_path: Path) -> Dict:
    """Process a single audio sample to extract voice characteristics"""
    logger.info(f"Processing audio sample: {sample_path}")
    
    try:
        # Load audio file
        y, sr = librosa.load(str(sample_path), sr=None)
        logger.info(f"Loaded audio with sample rate {sr}, duration: {len(y)/sr:.2f}s")
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            logger.info(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Extract basic features
        # Fundamental frequency estimation (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=70, fmax=300, fill_na=0.0, sr=sr
        )
        
        # Filter out unvoiced and unreliable segments
        reliable_f0 = f0[voiced_flag & (voiced_probs > 0.7)]
        if len(reliable_f0) > 0:
            mean_f0 = np.mean(reliable_f0)
            std_f0 = np.std(reliable_f0)
            logger.info(f"Mean F0: {mean_f0:.2f}Hz, Std F0: {std_f0:.2f}Hz")
        else:
            mean_f0 = 115.0  # Default for male voice
            std_f0 = 10.0
            logger.warning(f"Could not extract reliable F0, using defaults: {mean_f0}Hz")
        
        # Extract spectral features
        # Spectral centroid (brightness)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_cent = np.mean(cent)
        
        # Spectral contrast (formant structure)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        mean_contrast = np.mean(contrast, axis=1)
        
        # MFCC (timbre characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mean_mfccs = np.mean(mfccs, axis=1)
        
        # Voice characteristics dictionary
        voice_chars = {
            "fundamental_freq": float(mean_f0),
            "freq_std": float(std_f0),
            "spectral_centroid": float(mean_cent),
            "spectral_contrast": mean_contrast.tolist(),
            "mfccs": mean_mfccs.tolist(),
            "speaking_rate": 0.9,  # Default for Daniel Dennett
            "pitch_std": 1.2  # Default value for moderate pitch variation
        }
        
        # Create reference audio by taking a clean segment
        create_reference_audio(y, sr, sample_path.stem)
        
        return voice_chars
        
    except Exception as e:
        logger.error(f"Error processing sample {sample_path}: {str(e)}", exc_info=True)
        return {
            "fundamental_freq": 115.0,  # Default for male voice
            "freq_std": 10.0,
            "spectral_centroid": 2000.0,
            "spectral_contrast": [0.0] * 7,
            "mfccs": [0.0] * 13,
            "speaking_rate": 0.9,
            "pitch_std": 1.2
        }

def create_reference_audio(y: np.ndarray, sr: int, name: str):
    """Create a clean reference audio file"""
    try:
        # Find a segment with good energy (not silence)
        energy = librosa.feature.rms(y=y)[0]
        good_energy = energy > np.percentile(energy, 70)
        
        # Find a continuous segment of at least 3 seconds
        min_segment_length = int(3 * sr)
        segment_start = 0
        
        for i in range(len(good_energy) - min_segment_length):
            if np.all(good_energy[i:i+min_segment_length]):
                segment_start = i
                break
        
        segment_end = min(segment_start + min_segment_length, len(y))
        
        # Extract segment
        segment = y[segment_start:segment_end]
        
        # Normalize audio
        segment = segment / (np.max(np.abs(segment)) + 1e-8)
        
        # Save as reference audio
        output_path = REF_AUDIO_DIR / f"dennett_{name}_ref.wav"
        sf.write(str(output_path), segment, sr)
        logger.info(f"Created reference audio at {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating reference audio: {str(e)}", exc_info=True)

def create_voice_profile(voice_characteristics: List[Dict]):
    """Create a combined voice profile from multiple samples"""
    if not voice_characteristics:
        logger.error("No valid voice characteristics found to create profile")
        return
    
    try:
        logger.info(f"Creating voice profile from {len(voice_characteristics)} samples")
        
        # Average the fundamental frequencies and other numeric parameters
        f0_values = [vc["fundamental_freq"] for vc in voice_characteristics]
        mean_f0 = np.mean(f0_values)
        
        # Average MFCCs
        all_mfccs = np.array([vc["mfccs"] for vc in voice_characteristics])
        mean_mfccs = np.mean(all_mfccs, axis=0)
        
        # Create profile dictionary
        profile = {
            "fundamental_freq": float(mean_f0),
            "freq_std": float(np.mean([vc["freq_std"] for vc in voice_characteristics])),
            "spectral_centroid": float(np.mean([vc["spectral_centroid"] for vc in voice_characteristics])),
            "mfccs": mean_mfccs.tolist(),
            "speaking_rate": 0.9,  # Characteristic of Daniel Dennett
            "pitch_std": 1.2  # Moderate variation
        }
        
        # Save profile as JSON
        profile_json_path = PROFILE_DIR / "dennett_profile.json"
        with open(profile_json_path, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Saved voice profile to {profile_json_path}")
        
        # Create embedding tensor version (legacy format)
        # Convert profile to fixed-length embedding
        embedding = np.zeros(128)
        
        # Use the first 13 positions for MFCCs
        embedding[:13] = mean_mfccs
        
        # Add fundamental frequency
        embedding[13] = mean_f0 / 200.0  # Normalize
        
        # Add spectral features
        for i, vc in enumerate(voice_characteristics):
            if i < 5 and "spectral_contrast" in vc:
                start_idx = 14 + i * 7
                end_idx = start_idx + min(7, len(vc["spectral_contrast"]))
                embedding[start_idx:end_idx] = vc["spectral_contrast"][:end_idx-start_idx]
        
        # Randomize remaining values but seed with mean_f0 for deterministic results
        np.random.seed(int(mean_f0 * 10))
        remaining_indices = list(range(49, 128))
        embedding[remaining_indices] = np.random.randn(len(remaining_indices)) * 0.1
        
        # Normalize the embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Convert to tensor and save
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        profile_pt_path = PROFILE_DIR / "dennett_profile.pt"
        torch.save(embedding_tensor, profile_pt_path)
        logger.info(f"Saved voice embedding tensor to {profile_pt_path}")
        
    except Exception as e:
        logger.error(f"Error creating voice profile: {str(e)}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Process Daniel Dennett voice samples")
    parser.add_argument('--download', action='store_true', help="Download samples from the internet")
    args = parser.parse_args()
    
    # Prepare directories
    prepare_directories()
    
    # Find samples
    samples = find_voice_samples()
    
    if not samples:
        logger.warning("No voice samples found. Please add audio files to the data/voice_samples directory.")
        return
    
    # Process each sample
    voice_characteristics = []
    for sample in samples:
        chars = process_audio_sample(sample)
        if chars:
            voice_characteristics.append(chars)
    
    # Create combined voice profile
    create_voice_profile(voice_characteristics)
    
    logger.info("Voice processing complete!")

if __name__ == "__main__":
    main() 