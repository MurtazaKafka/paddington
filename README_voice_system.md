# Daniel Dennett Voice Synthesis System

This system automatically scrapes, processes, and creates a voice profile for philosopher Daniel Dennett, enabling the TTS system to generate speech that mimics his distinctive vocal characteristics.

## Components

1. **Advanced Dennett Scraper** (`advanced_dennett_scraper.py`): Collects audio samples of Daniel Dennett from various online sources including:
   - Philosophy Bites podcast episodes
   - Sean Carroll's Mindscape podcast
   - ABC Radio National interviews
   - YouTube lectures and interviews
   - Other podcast appearances

2. **Audio Processor** (`audio_processor.py`): Processes the downloaded audio files to:
   - Extract speech segments (removing music, silence, other speakers)
   - Normalize and enhance audio quality
   - Extract voice characteristics (pitch, tempo, timbre, etc.)
   - Generate a comprehensive voice profile

3. **Dennett Voice Adapter** (`dennett_voice_adapter.py`): Integrates with the TTS system to:
   - Apply Dennett's voice characteristics to speech generation
   - Customize speech synthesis parameters
   - Provide fallback parameters when a full profile isn't available

4. **Integration with ZonosTTS** (`zonos_tts.py`): Modified to use the voice adapter for more authentic speech generation.

5. **Workflow Script** (`run_dennett_voice_pipeline.sh`): Orchestrates the entire pipeline from downloading to processing.

## How to Use

1. Install the required dependencies:
   ```bash
   pip install -r requirements_scraper.txt
   pip install -r requirements_audio.txt
   ```

2. Run the full pipeline:
   ```bash
   ./run_dennett_voice_pipeline.sh
   ```

3. Alternatively, run each component separately:
   ```bash
   # Download samples
   python advanced_dennett_scraper.py --all
   
   # Process audio files
   python audio_processor.py
   
   # Copy the voice profile to the TTS system
   mkdir -p data/tts/voices
   cp data/voice_samples/voice_profile/dennett_voice_profile.json data/tts/voices/dennett.json
   ```

## Voice Characteristics Extracted

The system extracts and models the following voice characteristics:

- **Pitch**: Mean, standard deviation, range
- **Spectral Features**: Centroid, bandwidth
- **Timbre**: MFCCs (mel-frequency cepstral coefficients)
- **Rhythm**: Speaking rate, pauses
- **Energy**: Volume and dynamics

## Requirements

- Python 3.7+
- ffmpeg (for audio processing)
- yt-dlp (for YouTube downloads)
- espeak-ng (for phoneme conversion)

## Dependencies

See `requirements_scraper.txt` and `requirements_audio.txt` for detailed package requirements.

## References

- Daniel Dennett's voice samples primarily sourced from:
  - Philosophy Bites: https://philosophybites.com/
  - Sean Carroll's Mindscape: https://www.preposterousuniverse.com/podcast/
  - ABC Radio National: https://www.abc.net.au/radionational 