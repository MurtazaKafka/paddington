import os
import subprocess
import time
import requests
import json
from urllib.parse import urljoin

# Record duration in seconds
RECORD_DURATION = 5

# API base URL
BASE_URL = "https://127.0.0.1:8000"

def record_audio(output_file="recording.webm", duration=RECORD_DURATION):
    """Record audio from the microphone"""
    print(f"Recording audio for {duration} seconds...")
    
    # Command to record audio using ffmpeg
    cmd = [
        "ffmpeg",
        "-f", "pulse",  # Use PulseAudio
        "-i", "default",  # Use default input device
        "-t", str(duration),  # Duration in seconds
        "-c:a", "libopus",  # Use Opus codec
        "-b:a", "128k",  # Bitrate
        "-y",  # Overwrite output file if it exists
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Recording saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error recording audio: {e}")
        return False

def upload_audio(file_path):
    """Upload audio file to the server"""
    url = urljoin(BASE_URL, "upload-audio")
    
    try:
        # Open the file in binary mode
        with open(file_path, "rb") as f:
            # Create a multipart form with the file
            files = {"file": (os.path.basename(file_path), f, "audio/webm")}
            
            # Send the request
            print(f"Uploading {file_path} to {url}")
            response = requests.post(url, files=files, verify=False)
            
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Response received:")
                print(json.dumps(result, indent=2))
                
                # Check if audio was generated
                if result.get('audio_url'):
                    audio_url = result['audio_url']
                    print(f"Audio URL: {audio_url}")
                    
                    # Download the audio file
                    download_url = urljoin(BASE_URL, audio_url)
                    
                    print(f"Downloading response from {download_url}")
                    audio_response = requests.get(download_url, verify=False)
                    
                    if audio_response.status_code == 200:
                        # Save the audio file
                        output_path = "dennett_response.wav"
                        
                        with open(output_path, "wb") as f:
                            f.write(audio_response.content)
                        
                        print(f"Response saved to {output_path}")
                        print(f"File size: {os.path.getsize(output_path)} bytes")
                        
                        # Play the audio if possible
                        try:
                            print("Playing response...")
                            play_cmd = ["ffplay", "-nodisp", "-autoexit", output_path]
                            subprocess.run(play_cmd)
                        except:
                            print("Couldn't play audio automatically")
                            print(f"Use a media player to listen to {output_path}")
                    else:
                        print(f"Failed to download audio. Status code: {audio_response.status_code}")
                else:
                    print("No audio URL in the response")
                
                return result
            else:
                print(f"Error response: {response.text}")
                return None
    except Exception as e:
        print(f"Error uploading audio: {e}")
        return None

def main():
    """Main function to record and send audio"""
    # Suppress SSL warnings
    requests.packages.urllib3.disable_warnings()
    
    # Record audio
    recording_file = "recording.webm"
    if record_audio(recording_file):
        # Upload the recorded audio
        result = upload_audio(recording_file)
        
        if result:
            print("\nProcess completed successfully")
        else:
            print("\nProcess failed")
    else:
        print("Failed to record audio")

if __name__ == "__main__":
    main() 