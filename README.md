# Audio File Classifier

Classifies MP3 and M4A audio files into two categories:
- **English lyrics** - songs with detected English vocals
- **Other** - instrumentals, soundtracks, or non-English content

Uses OpenAI's Whisper model for speech recognition and language detection. Runs entirely locally - no API keys required.

## Requirements

- Python 3.10+
- ffmpeg

## Installation

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage - moves files into english/ and other/ subfolders
python mp3_classifier.py /path/to/audio/folder

# Preview without moving files
python mp3_classifier.py /path/to/audio/folder --dry-run

# Copy instead of move
python mp3_classifier.py /path/to/audio/folder --copy

# Use a more accurate model (slower)
python mp3_classifier.py /path/to/audio/folder --model small

# Analyze longer audio sample (default: 60 seconds)
python mp3_classifier.py /path/to/audio/folder --sample-duration 90

# Custom output folders
python mp3_classifier.py /path/to/audio/folder \
  --output-english /dest/english \
  --output-other /dest/other
```

## Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would happen without moving files |
| `--copy` | Copy files instead of moving them |
| `--model` | Whisper model: `tiny`, `base` (default), `small`, `medium`, `large` |
| `--sample-duration` | Seconds of audio to analyze (default: 60) |
| `--output-english` | Custom folder for English lyrics files |
| `--output-other` | Custom folder for non-English/instrumental files |

## How It Works

1. Extracts a sample from the middle of each audio file (to avoid intros/outros)
2. Converts to WAV format for Whisper compatibility
3. Transcribes audio and detects language using Whisper
4. Classifies as "English lyrics" if:
   - Language is detected as English
   - Transcription contains substantial text (20+ characters)
5. Moves/copies files to appropriate output folder

## Model Comparison

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | Fastest | Lower | ~1 GB |
| base | Fast | Good | ~1 GB |
| small | Medium | Better | ~2 GB |
| medium | Slow | High | ~5 GB |
| large | Slowest | Highest | ~10 GB |
