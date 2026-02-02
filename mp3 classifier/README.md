# Audio File Classifier

Classifies MP3 and M4A audio files into three categories:
- **songs/** - Files with English lyrics
- **ambiguous/** - Movie scenes, sparse dialogue, unclear content
- **other/** - Instrumentals, soundtracks, or non-English content

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
# Basic usage - sorts files into songs/, ambiguous/, and other/ subfolders
python mp3_classifier.py /path/to/audio/folder

# Preview without moving files
python mp3_classifier.py /path/to/audio/folder --dry-run

# Copy instead of move
python mp3_classifier.py /path/to/audio/folder --copy

# Use a more accurate model (slower)
python mp3_classifier.py /path/to/audio/folder --model small

# Adjust word density threshold
python mp3_classifier.py /path/to/audio/folder --min-wpm 40

# Custom output folders
python mp3_classifier.py /path/to/audio/folder \
  --output-songs /dest/songs \
  --output-ambiguous /dest/ambiguous \
  --output-other /dest/other
```

## Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would happen without moving files |
| `--copy` | Copy files instead of moving them |
| `--model` | Whisper model: `tiny`, `base` (default), `small`, `medium`, `large` |
| `--sample-duration` | Seconds per sample (default: 30). Three samples taken per file. |
| `--min-wpm` | Minimum words per minute to qualify as lyrics (default: 30) |
| `--output-songs` | Custom folder for songs with lyrics |
| `--output-ambiguous` | Custom folder for ambiguous files |
| `--output-other` | Custom folder for instrumental/non-English files |

## How It Works

The classifier uses a **multi-sample analysis** approach with three-category classification:

1. **Multi-position sampling**: Extracts 3 samples from different positions (25%, 50%, 75%) in each audio file
2. **Speech-to-text**: Converts each sample to WAV and transcribes using Whisper
3. **Word density check**: Calculates words-per-minute for each sample. Lyrics typically have 40-80 wpm, while sparse dialogue has 10-30 wpm.
4. **Three-category classification**:
   - **songs/**: 2+ samples pass (English detected + sufficient word density)
   - **ambiguous/**: 1 sample passes (some English content but inconsistent)
   - **other/**: 0 samples pass (instrumental or non-English)

This approach correctly handles:
- **Movie scenes**: Go to ambiguous/ (dialogue in only some sections)
- **Soundtracks with brief words**: Go to ambiguous/ (fails consistency check)
- **Songs with instrumental sections**: Go to songs/ (2+ samples still pass)
- **Pure instrumentals**: Go to other/ (no samples pass)

## Model Comparison

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | Fastest | Lower | ~1 GB |
| base | Fast | Good | ~1 GB |
| small | Medium | Better | ~2 GB |
| medium | Slow | High | ~5 GB |
| large | Slowest | Highest | ~10 GB |

## Tuning Tips

- **Too many songs in ambiguous/**: Decrease `--min-wpm` to 20-25
- **Too many movie scenes in songs/**: Increase `--min-wpm` to 40-50
- **Faster processing**: Use `--model tiny` (less accurate)
- **Better accuracy**: Use `--model small` or `--model medium`
