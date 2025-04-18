#!/usr/bin/env python3
"""
Advanced Daniel Dennett Audio Scraper

This script systematically searches and downloads various sources of Daniel Dennett's voice,
including podcasts, interviews, and lectures from multiple sources.
"""

import os
import sys
import logging
import requests
import json
import re
import time
import random
import subprocess
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dennett_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path("data/voice_samples")
DOWNLOAD_DIR = BASE_DIR / "downloads"
PROCESSED_DIR = BASE_DIR / "processed"
METADATA_FILE = BASE_DIR / "metadata.json"

# Hard-coded sources (manually curated)
SOURCES = {
    "philosophy_bites": {
        "name": "Philosophy Bites",
        "episodes": [
            {
                "title": "Daniel Dennett on Free Will Worth Wanting",
                "url": "http://traffic.libsyn.com/philosophybites/Daniel_Dennett_on_Free_Will_Worth_Wanting.mp3"
            },
            {
                "title": "Daniel Dennett on the Chinese Room",
                "url": "http://traffic.libsyn.com/philosophybites/Daniel_Dennett_on_the_Chinese_Room.mp3"
            },
            {
                "title": "Daniel Dennett on Intuition Pumps",
                "url": "http://traffic.libsyn.com/philosophybites/DennettIntPumps.mp3"
            }
        ]
    },
    "abc_radio": {
        "name": "ABC Radio National",
        "episodes": [
            {
                "title": "Free will, consciousness and AI: a conversation with Daniel Dennett",
                "url": "https://www.abc.net.au/listen/programs/philosopherszone/free-will-consciousness-ai-dennett/102911318",
                "type": "webpage"  # Needs special handling to extract audio
            }
        ]
    },
    "sean_carroll": {
        "name": "Sean Carroll's Mindscape",
        "episodes": [
            {
                "title": "Daniel Dennett on Minds, Patterns, and the Scientific Image",
                "url": "https://www.preposterousuniverse.com/podcast/2020/01/06/78-daniel-dennett-on-minds-patterns-and-the-scientific-image/",
                "type": "webpage"  # Needs special handling to extract audio
            }
        ]
    },
    "talks": {
        "name": "TED and Other Talks",
        "videos": [
            {
                "title": "The Illusion of Consciousness",
                "id": "fjbWv_QPWCM",
                "source": "youtube"
            },
            {
                "title": "Daniel Dennett: The Genius of Charles Darwin",
                "id": "G4Yhn2uQiog",
                "source": "youtube"
            },
            {
                "title": "Tools To Transform Our Thinking",
                "id": "EJsD-3jtXz0",
                "source": "youtube"
            }
        ]
    }
}

# YouTube playlist IDs containing Daniel Dennett content
YOUTUBE_PLAYLISTS = [
    "PLKWKTPomEmKZU5_Oc7v0kRrX2xDWMYyq5",  # Daniel Dennett Lectures
    "PLAYlNiRs-J2qLBPSfisZJ-s47k-74z-o5"   # Daniel Dennett Interviews
]

# Helper functions
def ensure_directories():
    """Create necessary directories"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info(f"Created directories: {DOWNLOAD_DIR}, {PROCESSED_DIR}")

def load_metadata():
    """Load metadata of already downloaded files"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"downloads": [], "last_updated": None}

def save_metadata(metadata):
    """Save metadata to file"""
    metadata["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Saved metadata: {len(metadata['downloads'])} entries")

def is_already_downloaded(url, metadata):
    """Check if a URL has already been downloaded"""
    for entry in metadata["downloads"]:
        if entry["url"] == url:
            return True
    return False

def download_direct_audio(url, filename, metadata):
    """Download audio directly from a URL"""
    if is_already_downloaded(url, metadata):
        logger.info(f"Already downloaded: {filename}")
        return None
    
    try:
        logger.info(f"Downloading audio: {filename}")
        # Add timeout to avoid hanging downloads
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code == 200:
            file_path = DOWNLOAD_DIR / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded: {file_path}")
            
            # Add to metadata
            metadata["downloads"].append({
                "url": url,
                "filename": str(file_path),
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": os.path.getsize(file_path)
            })
            save_metadata(metadata)
            
            return file_path
        else:
            logger.error(f"Failed to download {url}: Status code {response.status_code}")
            return None
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout downloading {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None

def download_youtube_audio(video_id, title, metadata):
    """Download audio from YouTube using yt-dlp"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    if is_already_downloaded(url, metadata):
        logger.info(f"Already downloaded: {title}")
        return None
    
    try:
        logger.info(f"Downloading YouTube audio: {title} ({video_id})")
        output_filename = re.sub(r'[^\w\-\.]', '_', title)
        output_path = DOWNLOAD_DIR / f"{output_filename}.wav"
        
        # Use yt-dlp to download audio
        cmd = [
            "yt-dlp",
            "-x",                     # Extract audio
            "--audio-format", "wav",  # Convert to WAV format
            "--audio-quality", "0",   # Best quality
            "-o", str(output_path),   # Output path
            url
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if os.path.exists(output_path):
            logger.info(f"Successfully downloaded: {output_path}")
            
            # Add to metadata
            metadata["downloads"].append({
                "url": url,
                "filename": str(output_path),
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": os.path.getsize(output_path),
                "source": "youtube",
                "video_id": video_id
            })
            save_metadata(metadata)
            
            return output_path
        else:
            logger.error(f"Failed to download YouTube audio for {video_id}")
            return None
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading YouTube audio for {video_id}: {e}")
        logger.error(f"Command output: {e.stdout}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio: {e}")
        return None

def get_youtube_playlist_videos(playlist_id):
    """Get all video IDs and titles from a YouTube playlist"""
    try:
        logger.info(f"Fetching videos from YouTube playlist: {playlist_id}")
        
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print", "id,title",
            f"https://www.youtube.com/playlist?list={playlist_id}"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if ' ' in line:
                video_id, title = line.split(' ', 1)
                # Filter for Daniel Dennett content only
                if 'dennett' in title.lower():
                    videos.append({
                        "id": video_id,
                        "title": title,
                        "source": "youtube"
                    })
        
        logger.info(f"Found {len(videos)} Dennett videos in playlist {playlist_id}")
        return videos
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching YouTube playlist {playlist_id}: {e}")
        logger.error(f"Command output: {e.stdout}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching YouTube playlist: {e}")
        return []

def extract_audio_from_webpage(url, source_name):
    """Extract audio URL from a webpage"""
    try:
        logger.info(f"Extracting audio URL from webpage: {url}")
        
        # Add timeout to the request
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Different extraction strategies for different sites
        audio_url = None
        
        if "abc.net.au" in url:
            # ABC Radio National
            audio_elements = soup.select('audio source[src]')
            for element in audio_elements:
                src = element.get('src')
                if src and (src.endswith('.mp3') or src.endswith('.m4a')):
                    audio_url = src
                    break
                    
        elif "preposterousuniverse.com" in url:
            # Sean Carroll's Mindscape
            audio_elements = soup.select('audio[src], a[href$=".mp3"]')
            for element in audio_elements:
                src = element.get('src') or element.get('href')
                if src and (src.endswith('.mp3') or src.endswith('.m4a')):
                    audio_url = src
                    break
                    
            # If not found, try embedded iframe
            if not audio_url:
                iframes = soup.select('iframe[src*="player.simplecast.com"]')
                if iframes:
                    iframe_url = iframes[0].get('src')
                    try:
                        iframe_response = requests.get(iframe_url, timeout=30)
                        iframe_soup = BeautifulSoup(iframe_response.text, 'html.parser')
                        audio_elements = iframe_soup.select('audio[src]')
                        for element in audio_elements:
                            src = element.get('src')
                            if src:
                                audio_url = src
                                break
                    except (requests.exceptions.RequestException, Exception) as e:
                        logger.error(f"Error fetching iframe content: {e}")
        
        # Make sure URL is absolute
        if audio_url and not audio_url.startswith(('http://', 'https://')):
            audio_url = urljoin(url, audio_url)
            
        if audio_url:
            logger.info(f"Found audio URL: {audio_url}")
            return audio_url
        else:
            logger.warning(f"Could not extract audio URL from {url}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout extracting audio from {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error extracting audio from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting audio from webpage {url}: {e}")
        return None

def search_for_dennett_podcasts():
    """Search for additional Daniel Dennett podcast episodes"""
    known_podcasts = [
        # Very Bad Wizards
        {
            "name": "Very Bad Wizards",
            "base_url": "https://verybadwizards.com/episodes",
            "search_term": "dennett"
        },
        # Making Sense with Sam Harris
        {
            "name": "Making Sense with Sam Harris",
            "base_url": "https://www.samharris.org/podcasts/making-sense-episodes",
            "search_term": "dennett"
        },
        # On Being
        {
            "name": "On Being",
            "base_url": "https://onbeing.org/libraries/",
            "search_term": "daniel dennett"
        }
    ]
    
    results = []
    
    for podcast in known_podcasts:
        try:
            logger.info(f"Searching for Daniel Dennett episodes in {podcast['name']}")
            response = requests.get(f"{podcast['base_url']}", timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Search for episodes with Dennett in the title
                episode_elements = soup.select('a[href*="episode"], a[href*="episodes"], h2, h3, .episode-title')
                
                for element in episode_elements:
                    text = element.get_text().lower()
                    if podcast['search_term'].lower() in text:
                        # Found a potential match
                        episode_url = element.get('href')
                        if not episode_url and element.find_parent('a'):
                            episode_url = element.find_parent('a').get('href')
                            
                        if episode_url:
                            if not episode_url.startswith(('http://', 'https://')):
                                episode_url = urljoin(podcast['base_url'], episode_url)
                                
                            results.append({
                                "title": element.get_text().strip(),
                                "url": episode_url,
                                "source": podcast['name'],
                                "type": "webpage"
                            })
            
            logger.info(f"Found {len(results)} potential episodes in {podcast['name']}")
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout searching {podcast['name']}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error searching {podcast['name']}: {e}")
        except Exception as e:
            logger.error(f"Error searching {podcast['name']}: {e}")
    
    return results

def process_all_sources(metadata, max_downloads=5):
    """Process all known sources and download audio"""
    total_downloads = 0
    
    # Process hard-coded direct audio URLs
    for source_key, source_data in SOURCES.items():
        if source_key == "philosophy_bites":
            for episode in source_data["episodes"]:
                if total_downloads >= max_downloads and max_downloads > 0:
                    logger.info(f"Reached maximum download limit ({max_downloads})")
                    return total_downloads
                    
                filename = f"philosophy_bites_{episode['title'].replace(' ', '_')}.mp3"
                result = download_direct_audio(episode["url"], filename, metadata)
                if result:
                    total_downloads += 1
                time.sleep(random.uniform(1.0, 3.0))  # Be nice to servers
                
        elif source_key == "talks":
            for video in source_data["videos"]:
                if total_downloads >= max_downloads and max_downloads > 0:
                    logger.info(f"Reached maximum download limit ({max_downloads})")
                    return total_downloads
                    
                result = download_youtube_audio(video["id"], video["title"], metadata)
                if result:
                    total_downloads += 1
                time.sleep(random.uniform(1.0, 3.0))
                
        elif "episodes" in source_data:
            for episode in source_data["episodes"]:
                if total_downloads >= max_downloads and max_downloads > 0:
                    logger.info(f"Reached maximum download limit ({max_downloads})")
                    return total_downloads
                    
                if episode.get("type") == "webpage":
                    # Extract audio URL from webpage
                    audio_url = extract_audio_from_webpage(episode["url"], source_data["name"])
                    if audio_url:
                        filename = f"{source_key}_{episode['title'].replace(' ', '_')}.mp3"
                        result = download_direct_audio(audio_url, filename, metadata)
                        if result:
                            total_downloads += 1
                    time.sleep(random.uniform(2.0, 5.0))
    
    # If we've reached the max downloads, return early
    if total_downloads >= max_downloads and max_downloads > 0:
        logger.info(f"Reached maximum download limit ({max_downloads})")
        return total_downloads
    
    # Process YouTube playlists
    for playlist_id in YOUTUBE_PLAYLISTS:
        videos = get_youtube_playlist_videos(playlist_id)
        for video in videos:
            if total_downloads >= max_downloads and max_downloads > 0:
                logger.info(f"Reached maximum download limit ({max_downloads})")
                return total_downloads
                
            result = download_youtube_audio(video["id"], video["title"], metadata)
            if result:
                total_downloads += 1
            time.sleep(random.uniform(1.0, 3.0))
    
    # If we've reached the max downloads, return early
    if total_downloads >= max_downloads and max_downloads > 0:
        logger.info(f"Reached maximum download limit ({max_downloads})")
        return total_downloads
    
    # Search for additional podcast episodes
    additional_episodes = search_for_dennett_podcasts()
    for episode in additional_episodes:
        if total_downloads >= max_downloads and max_downloads > 0:
            logger.info(f"Reached maximum download limit ({max_downloads})")
            return total_downloads
            
        if episode.get("type") == "webpage":
            audio_url = extract_audio_from_webpage(episode["url"], episode["source"])
            if audio_url:
                filename = f"{episode['source'].replace(' ', '_')}_{episode['title'].replace(' ', '_')}.mp3"
                result = download_direct_audio(audio_url, filename, metadata)
                if result:
                    total_downloads += 1
            time.sleep(random.uniform(2.0, 5.0))
    
    return total_downloads

def main():
    """Main function to download Daniel Dennett audio"""
    parser = argparse.ArgumentParser(description="Download Daniel Dennett audio samples")
    parser.add_argument('--all', action='store_true', help="Download from all sources")
    parser.add_argument('--youtube', action='store_true', help="Download YouTube samples only")
    parser.add_argument('--podcasts', action='store_true', help="Download podcast samples only")
    parser.add_argument('--max', type=int, default=5, help="Maximum number of files to download (0 for unlimited)")
    
    try:
        args = parser.parse_args()
    except ImportError:
        # Simplified argument parsing if argparse is not available
        class Args:
            def __init__(self):
                self.all = True
                self.youtube = False
                self.podcasts = False
                self.max = 5
        args = Args()
    
    # Create necessary directories
    ensure_directories()
    
    # Load metadata
    metadata = load_metadata()
    
    logger.info("Starting Daniel Dennett audio scraper")
    logger.info(f"Found {len(metadata['downloads'])} previously downloaded files")
    logger.info(f"Maximum downloads per run: {args.max}")
    
    # Start downloading
    try:
        if args.all or (not args.youtube and not args.podcasts):
            total_downloads = process_all_sources(metadata, args.max)
        elif args.youtube:
            # Download only from YouTube sources
            total_downloads = 0
            max_remaining = args.max
            
            for source_key, source_data in SOURCES.items():
                if source_key == "talks":
                    for video in source_data["videos"]:
                        if total_downloads >= args.max and args.max > 0:
                            break
                        
                        result = download_youtube_audio(video["id"], video["title"], metadata)
                        if result:
                            total_downloads += 1
                        time.sleep(random.uniform(1.0, 3.0))
            
            if total_downloads < args.max or args.max == 0:
                for playlist_id in YOUTUBE_PLAYLISTS:
                    if total_downloads >= args.max and args.max > 0:
                        break
                    
                    videos = get_youtube_playlist_videos(playlist_id)
                    for video in videos:
                        if total_downloads >= args.max and args.max > 0:
                            break
                        
                        result = download_youtube_audio(video["id"], video["title"], metadata)
                        if result:
                            total_downloads += 1
                        time.sleep(random.uniform(1.0, 3.0))
        elif args.podcasts:
            # Download only from podcast sources
            total_downloads = 0
            
            for source_key, source_data in SOURCES.items():
                if source_key in ["philosophy_bites", "abc_radio", "sean_carroll"]:
                    for episode in source_data["episodes"]:
                        if total_downloads >= args.max and args.max > 0:
                            break
                        
                        if "type" in episode and episode["type"] == "webpage":
                            audio_url = extract_audio_from_webpage(episode["url"], source_data["name"])
                            if audio_url:
                                filename = f"{source_key}_{episode['title'].replace(' ', '_')}.mp3"
                                result = download_direct_audio(audio_url, filename, metadata)
                                if result:
                                    total_downloads += 1
                        else:
                            filename = f"{source_key}_{episode['title'].replace(' ', '_')}.mp3"
                            result = download_direct_audio(episode["url"], filename, metadata)
                            if result:
                                total_downloads += 1
                        time.sleep(random.uniform(1.0, 3.0))
                        
                    if total_downloads >= args.max and args.max > 0:
                        break
            
            if total_downloads < args.max or args.max == 0:
                additional_episodes = search_for_dennett_podcasts()
                for episode in additional_episodes:
                    if total_downloads >= args.max and args.max > 0:
                        break
                    
                    if episode.get("type") == "webpage":
                        audio_url = extract_audio_from_webpage(episode["url"], episode["source"])
                        if audio_url:
                            filename = f"{episode['source'].replace(' ', '_')}_{episode['title'].replace(' ', '_')}.mp3"
                            result = download_direct_audio(audio_url, filename, metadata)
                            if result:
                                total_downloads += 1
                    time.sleep(random.uniform(2.0, 5.0))
        
        logger.info(f"Download complete! Downloaded {total_downloads} new files")
        logger.info(f"Total files in collection: {len(metadata['downloads'])}")
    
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        save_metadata(metadata)

if __name__ == "__main__":
    main() 