#!/usr/bin/env python3
"""
Download Daniel Dennett audio samples for voice synthesis.
This script helps gather audio samples from various sources.
"""

import os
import sys
import logging
import requests
import subprocess
from pathlib import Path
import argparse
import re
import time
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths
SAMPLES_DIR = Path("data/voice_samples")

# Daniel Dennett YouTube lecture and interview IDs
DENNETT_VIDEOS = [
    # Big Think interviews
    {"id": "JP1nmExfgpg", "title": "dennett_consciousness_explained"},
    {"id": "vQkd0j-EK_k", "title": "dennett_consciousness_evolved"}, 
    {"id": "5fyjcMscofM", "title": "dennett_free_will"},
    
    # Lectures
    {"id": "A2VZ6E6zmMw", "title": "dennett_what_can_cognitive_science"},
    {"id": "CZGl1SRYvoQ", "title": "dennett_consciousness_illusion"},
    
    # Interviews
    {"id": "h0-04ia-h-Q", "title": "dennett_interview_philosophy_bites"},
    {"id": "OdNvrwH-eAM", "title": "dennett_sam_harris_discussion"}
]

def prepare_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    logger.info(f"Created/verified directory: {SAMPLES_DIR}")

def download_youtube_audio(video_info: dict):
    """Download audio from YouTube video"""
    video_id = video_info["id"]
    title = video_info["title"]
    output_path = SAMPLES_DIR / f"{title}.wav"
    
    if output_path.exists():
        logger.info(f"File already exists: {output_path}, skipping download")
        return True
    
    try:
        logger.info(f"Downloading audio for video: {video_id} - {title}")
        
        # Use yt-dlp to download audio (better maintained than youtube-dl)
        cmd = [
            "yt-dlp",
            "-x",                     # Extract audio
            "--audio-format", "wav",  # Convert to WAV format
            "--audio-quality", "0",   # Best quality
            "-o", f"{output_path}",   # Output path
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if output_path.exists():
            logger.info(f"Successfully downloaded audio to {output_path}")
            return True
        else:
            logger.error(f"Failed to download audio for video {video_id}. Command output: {result.stdout}")
            return False
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading YouTube audio for {video_id}: {e}")
        logger.error(f"Command output: {e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio: {e}")
        return False

def find_and_download_dennett_podcasts():
    """Find and download podcasts featuring Daniel Dennett"""
    
    # Example podcast URLs (manually collected)
    dennett_podcasts = [
        {
            "url": "https://traffic.libsyn.com/secure/verybadwizards/VBW_103_PD4.mp3",
            "title": "dennett_very_bad_wizards_interview"
        },
        {
            "url": "https://traffic.libsyn.com/secure/samharris/Making_Sense_Ep267-SIMULCAST.mp3", 
            "title": "dennett_making_sense_interview"
        }
    ]
    
    for podcast in dennett_podcasts:
        output_path = SAMPLES_DIR / f"{podcast['title']}.mp3"
        
        if output_path.exists():
            logger.info(f"Podcast already downloaded: {output_path}")
            continue
            
        try:
            logger.info(f"Downloading podcast: {podcast['title']}")
            response = requests.get(podcast['url'], stream=True)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                logger.info(f"Successfully downloaded podcast to {output_path}")
            else:
                logger.error(f"Failed to download podcast, status code: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error downloading podcast {podcast['title']}: {e}")

def main():
    """Main function to download Daniel Dennett audio samples"""
    parser = argparse.ArgumentParser(description="Download Daniel Dennett audio samples")
    parser.add_argument('--all', action='store_true', help="Download all available samples")
    parser.add_argument('--youtube', action='store_true', help="Download YouTube samples")
    parser.add_argument('--podcasts', action='store_true', help="Download podcast samples")
    
    args = parser.parse_args()
    
    # Default to all if no specific option is selected
    if not (args.youtube or args.podcasts):
        args.all = True
        
    # Prepare directories
    prepare_directories()
    
    # Download from specified sources
    if args.all or args.youtube:
        logger.info("Downloading YouTube samples...")
        for video in DENNETT_VIDEOS:
            download_youtube_audio(video)
            # Add delay between downloads to avoid rate limiting
            time.sleep(random.uniform(1.0, 3.0))
    
    if args.all or args.podcasts:
        logger.info("Downloading podcast samples...")
        find_and_download_dennett_podcasts()
    
    logger.info("Download process complete!")

if __name__ == "__main__":
    main() 