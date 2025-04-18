#!/usr/bin/env python3
"""
Daniel Dennett Audio Processor

This script processes downloaded Daniel Dennett audio files to:
1. Extract speech segments (removing music, silence, other speakers)
2. Normalize and enhance audio quality
3. Extract voice characteristics for TTS training
"""

import os
import sys
import glob
import json
import logging
import subprocess
import time
from pathlib import Path
import numpy as np
import argparse
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("audio_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path("data/voice_samples")
DOWNLOAD_DIR = BASE_DIR / "downloads"
PROCESSED_DIR = BASE_DIR / "processed"
VOICE_PROFILE_DIR = BASE_DIR / "voice_profile"
METADATA_FILE = BASE_DIR / "metadata.json"
VOICE_PROFILE_FILE = VOICE_PROFILE_DIR / "dennett_voice_profile.json"

def ensure_directories():
    """Create necessary directories"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VOICE_PROFILE_DIR, exist_ok=True)
    logger.info(f"Created directories: {PROCESSED_DIR}, {VOICE_PROFILE_DIR}")

def convert_audio_to_wav(input_file, output_file, sample_rate=22050):
    """Convert any audio format to WAV with consistent parameters"""
    try:
        logger.info(f"Converting {input_file} to {output_file}")
        if input_file.endswith('.mp3'):
            # Use pydub for mp3 conversion
            audio = AudioSegment.from_mp3(input_file)
            audio = audio.set_frame_rate(sample_rate)
            audio = audio.set_channels(1)  # Mono
            audio.export(output_file, format="wav")
        else:
            # Use librosa for other formats
            y, sr = librosa.load(input_file, sr=sample_rate, mono=True)
            sf.write(output_file, y, sample_rate)
        logger.info(f"Successfully converted {input_file}")
        return True
    except Exception as e:
        logger.error(f"Error converting {input_file}: {e}")
        return False

def extract_speech_segments(input_file, output_dir, min_silence_len=500, silence_thresh=-40):
    """Split audio on silence and extract likely speech segments"""
    try:
        logger.info(f"Extracting speech segments from {input_file}")
        filename = os.path.basename(input_file)
        base_name = os.path.splitext(filename)[0]
        
        # Load audio
        audio = AudioSegment.from_wav(input_file)
        
        # Split on silence
        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=300  # Keep 300ms of silence on each side
        )
        
        logger.info(f"Found {len(segments)} speech segments in {input_file}")
        
        # Save segments that are likely speech (between 1 and 30 seconds)
        segment_paths = []
        for i, segment in enumerate(segments):
            # Filter segments by duration (1-30 seconds)
            if 1000 <= len(segment) <= 30000:
                segment_path = os.path.join(output_dir, f"{base_name}_segment_{i:03d}.wav")
                segment.export(segment_path, format="wav")
                segment_paths.append(segment_path)
        
        logger.info(f"Saved {len(segment_paths)} usable speech segments")
        return segment_paths
    
    except Exception as e:
        logger.error(f"Error extracting speech from {input_file}: {e}")
        return []

def extract_voice_features(audio_file):
    """Extract voice features from an audio file"""
    try:
        logger.info(f"Extracting voice features from {audio_file}")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        
        # Extract features
        features = {}
        
        # Pitch (F0) statistics
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[magnitudes > np.median(magnitudes)]
        pitches = pitches[pitches > 0]  # Remove zeros
        
        if len(pitches) > 0:
            features["pitch_mean"] = float(np.mean(pitches))
            features["pitch_std"] = float(np.std(pitches))
            features["pitch_min"] = float(np.min(pitches))
            features["pitch_max"] = float(np.max(pitches))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfccs"] = [float(np.mean(mfcc)) for mfcc in mfccs]
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo)
        
        # Energy features
        features["energy_mean"] = float(np.mean(np.abs(y)))
        features["energy_std"] = float(np.std(np.abs(y)))
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))
        
        logger.info(f"Successfully extracted features from {audio_file}")
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features from {audio_file}: {e}")
        return {}

def create_voice_profile(processed_files):
    """Create a voice profile from processed audio files"""
    try:
        logger.info(f"Creating voice profile from {len(processed_files)} files")
        
        all_features = []
        for audio_file in processed_files:
            features = extract_voice_features(audio_file)
            if features:
                all_features.append(features)
        
        if not all_features:
            logger.error("No valid features extracted, cannot create voice profile")
            return None
        
        # Get current time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Aggregate features
        profile = {
            "voice_id": "daniel_dennett",
            "name": "Daniel Dennett",
            "description": "Voice profile of philosopher Daniel Dennett",
            "created_at": current_time,
            "sample_count": len(all_features),
            "features": {}
        }
        
        # Calculate average features
        for key in all_features[0].keys():
            if key == "mfccs":
                # Handle MFCCs separately (list of values)
                mfcc_values = [f["mfccs"] for f in all_features]
                avg_mfccs = [sum(values) / len(values) for values in zip(*mfcc_values)]
                profile["features"]["mfccs"] = avg_mfccs
            else:
                # Handle scalar features
                values = [f[key] for f in all_features if key in f]
                if values:
                    profile["features"][key] = sum(values) / len(values)
        
        # Save profile
        os.makedirs(os.path.dirname(VOICE_PROFILE_FILE), exist_ok=True)
        with open(VOICE_PROFILE_FILE, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Voice profile created and saved to {VOICE_PROFILE_FILE}")
        return profile
    
    except Exception as e:
        logger.error(f"Error creating voice profile: {e}")
        return None

def process_all_audio():
    """Process all downloaded audio files"""
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Get all downloaded audio files
        audio_files = []
        for ext in ['.mp3', '.wav', '.m4a', '.aac', '.ogg']:
            audio_files.extend(glob.glob(os.path.join(DOWNLOAD_DIR, f"*{ext}")))
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        processed_files = []
        for audio_file in audio_files:
            # Convert to standard WAV format if needed
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            temp_wav = os.path.join(PROCESSED_DIR, f"{base_name}_temp.wav")
            
            if not audio_file.endswith('.wav'):
                if not convert_audio_to_wav(audio_file, temp_wav):
                    continue
            else:
                temp_wav = audio_file
            
            # Extract speech segments
            segments = extract_speech_segments(temp_wav, PROCESSED_DIR)
            processed_files.extend(segments)
            
            # Remove temporary WAV if we created one
            if temp_wav != audio_file and os.path.exists(temp_wav):
                os.remove(temp_wav)
        
        logger.info(f"Processed {len(processed_files)} speech segments")
        
        # Create voice profile
        if processed_files:
            profile = create_voice_profile(processed_files)
            if profile:
                logger.info("Voice profile created successfully")
            else:
                logger.error("Failed to create voice profile")
        else:
            logger.warning("No processed files available for voice profile creation")
        
        return processed_files
    
    except Exception as e:
        logger.error(f"Error in audio processing: {e}")
        return []

def main():
    """Main function"""
    import time
    
    parser = argparse.ArgumentParser(description="Process Daniel Dennett audio samples")
    parser.add_argument('--force', action='store_true', help="Force reprocessing of all files")
    
    try:
        args = parser.parse_args()
    except ImportError:
        # Simplified argument parsing if argparse is not available
        class Args:
            def __init__(self):
                self.force = False
        args = Args()
    
    logger.info("Starting Daniel Dennett audio processor")
    
    # Process all audio
    processed_files = process_all_audio()
    
    logger.info(f"Audio processing complete! Processed {len(processed_files)} files")

if __name__ == "__main__":
    main() 