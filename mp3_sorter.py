import os
import random
import shutil
import sys
from pathlib import Path

import whisper
from pydub import AudioSegment
from tqdm import tqdm


# Configuration
AUDIO_DIR = "./audio_files"
HAS_LYRICS_DIR = "./Has-lyrics"
LYRICLESS_DIR = "./lyricless"
UNSURE_DIR = "./unsure"

NUM_SAMPLES = 4
SAMPLE_DURATION_MS = 10000  # 10 seconds per sample
EXCLUDE_PERCENT = 0.10  # Exclude first and last 10%
LYRICS_THRESHOLD = 2  # Need 2+ samples with speech to classify as "Has-lyrics"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large


def ensure_directories():
    """Create output directories if they don't exist."""
    for dir_path in [HAS_LYRICS_DIR, LYRICLESS_DIR, UNSURE_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_random_sample_positions(duration_ms: int, num_samples: int) -> list[int]:
    """
    Get random sample start positions, excluding first and last 10% of the file.
    """
    # Calculate the valid range (exclude first and last 10%)
    start_boundary = int(duration_ms * EXCLUDE_PERCENT)
    end_boundary = int(duration_ms * (1 - EXCLUDE_PERCENT))

    # Ensure we have enough room for samples
    valid_range = end_boundary - start_boundary - SAMPLE_DURATION_MS
    if valid_range <= 0:
        # File too short, just sample from middle
        middle = duration_ms // 2
        return [max(0, middle - SAMPLE_DURATION_MS // 2)]

    # Generate random positions within valid range
    positions = []
    for _ in range(num_samples):
        pos = random.randint(start_boundary, end_boundary - SAMPLE_DURATION_MS)
        positions.append(pos)

    return sorted(positions)


def check_sample_for_speech(audio_segment: AudioSegment, model) -> str:
    """
    Check if an audio sample contains speech using Whisper.
    Returns: "speech", "no_speech", or "ambiguous"
    """
    # Export to temporary WAV for Whisper
    temp_path = "/tmp/temp_sample.wav"
    audio_segment.export(temp_path, format="wav")

    try:
        # Transcribe with Whisper
        result = model.transcribe(temp_path, language="en", fp16=False)
        text = result.get("text", "").strip()

        # If we got text back, there's speech
        if text and len(text) > 0:
            return "speech"
        return "no_speech"

    except Exception as e:
        print(f"    Whisper error: {e}")
        return "ambiguous"
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def classify_audio_file(file_path: str, model) -> str:
    """
    Classify an audio file based on speech detection in random samples.
    Returns: "has_lyrics", "lyricless", or "unsure"
    """
    try:
        # Load the audio file
        ext = Path(file_path).suffix.lower()
        if ext == ".mp3":
            audio = AudioSegment.from_mp3(file_path)
        elif ext == ".m4a":
            audio = AudioSegment.from_file(file_path, format="m4a")
        elif ext == ".wav":
            audio = AudioSegment.from_wav(file_path)
        else:
            audio = AudioSegment.from_file(file_path)

        duration_ms = len(audio)

        # Get random sample positions
        positions = get_random_sample_positions(duration_ms, NUM_SAMPLES)

        # Analyze each sample
        results = {"speech": 0, "no_speech": 0, "ambiguous": 0}

        for pos in positions:
            end_pos = min(pos + SAMPLE_DURATION_MS, duration_ms)
            sample = audio[pos:end_pos]
            result = check_sample_for_speech(sample, model)
            results[result] += 1

        # Classify based on results
        if results["ambiguous"] > NUM_SAMPLES // 2:
            return "unsure"
        elif results["speech"] >= LYRICS_THRESHOLD:
            return "has_lyrics"
        else:
            return "lyricless"

    except Exception as e:
        print(f"    Error processing file: {e}")
        return "unsure"


def move_file(file_path: str, classification: str):
    """Move file to appropriate directory based on classification."""
    filename = os.path.basename(file_path)

    if classification == "has_lyrics":
        dest = os.path.join(HAS_LYRICS_DIR, filename)
    elif classification == "lyricless":
        dest = os.path.join(LYRICLESS_DIR, filename)
    else:
        dest = os.path.join(UNSURE_DIR, filename)

    shutil.move(file_path, dest)
    return dest


def main():
    ensure_directories()

    # Get all audio files
    audio_extensions = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
    files = [
        f for f in os.listdir(AUDIO_DIR)
        if Path(f).suffix.lower() in audio_extensions
    ]

    print(f"Found {len(files)} audio files to process")

    if not files:
        print(f"No audio files found in {AUDIO_DIR} directory")
        return

    # Load Whisper model
    print(f"Loading Whisper model ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)

    # Track results
    counts = {"has_lyrics": 0, "lyricless": 0, "unsure": 0}

    for filename in tqdm(files, desc="Processing audio files"):
        file_path = os.path.join(AUDIO_DIR, filename)

        # Classify the file
        classification = classify_audio_file(file_path, model)
        counts[classification] += 1

        # Move to appropriate folder
        move_file(file_path, classification)

    # Print summary (flush to ensure proper ordering after tqdm)
    sys.stderr.flush()
    sys.stdout.flush()
    print("\n=== Classification Summary ===")
    print(f"Has-lyrics: {counts['has_lyrics']}")
    print(f"Lyricless:  {counts['lyricless']}")
    print(f"Unsure:     {counts['unsure']}")


if __name__ == "__main__":
    main()
