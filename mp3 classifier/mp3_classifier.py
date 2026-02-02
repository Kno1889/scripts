#!/usr/bin/env python3
"""
Audio File Classifier
Classifies MP3 and M4A files into three categories:
- songs: Files with English lyrics (2+ samples pass)
- ambiguous: Movie scenes, sparse dialogue (1 sample passes)
- other: Instrumental or non-English (0 samples pass)

Uses OpenAI's Whisper model for speech recognition and language detection.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Error: whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub not installed. Run: pip install pydub")
    sys.exit(1)

from tqdm import tqdm


def load_audio(audio_path: Path) -> AudioSegment:
    """Load audio file based on its format."""
    suffix = audio_path.suffix.lower()
    if suffix == '.mp3':
        return AudioSegment.from_mp3(audio_path)
    elif suffix == '.m4a':
        return AudioSegment.from_file(audio_path, format='m4a')
    else:
        return AudioSegment.from_file(audio_path)


def extract_audio_sample(
    audio: AudioSegment,
    audio_path: Path,
    sample_duration_ms: int,
    position: float,
    sample_index: int = 0
) -> tuple[Path, int]:
    """
    Extract a sample from a specific position in the audio.

    Args:
        audio: Loaded AudioSegment
        audio_path: Original file path (for temp file naming)
        sample_duration_ms: Duration of sample in milliseconds
        position: Position as fraction (0.0 = start, 0.5 = middle, 1.0 = end)
        sample_index: Index for temp file naming

    Returns:
        Tuple of (temp_wav_path, actual_duration_ms)
    """
    duration = len(audio)

    if duration <= sample_duration_ms:
        sample = audio
        actual_duration = duration
    else:
        # Calculate start position, ensuring we don't go past the end
        max_start = duration - sample_duration_ms
        start = int(max_start * position)
        sample = audio[start:start + sample_duration_ms]
        actual_duration = sample_duration_ms

    # Export as WAV for Whisper
    temp_path = audio_path.with_suffix(f'.temp{sample_index}.wav')
    sample.export(temp_path, format='wav')
    return temp_path, actual_duration


def extract_multiple_samples(
    audio_path: Path,
    sample_duration_ms: int = 30000,
    positions: list[float] = None
) -> list[tuple[Path, int]]:
    """
    Extract multiple samples from different positions in the audio.

    Args:
        audio_path: Path to audio file
        sample_duration_ms: Duration of each sample
        positions: List of positions as fractions (default: [0.25, 0.5, 0.75])

    Returns:
        List of (temp_wav_path, actual_duration_ms) tuples
    """
    if positions is None:
        positions = [0.25, 0.5, 0.75]

    audio = load_audio(audio_path)
    samples = []

    for i, pos in enumerate(positions):
        temp_path, duration = extract_audio_sample(
            audio, audio_path, sample_duration_ms, pos, i
        )
        samples.append((temp_path, duration))

    return samples


def analyze_audio(model, audio_path: Path) -> dict:
    """
    Analyze audio file using Whisper to detect language and transcribe.
    Returns dict with 'language', 'confidence', and 'text'.
    """
    result = model.transcribe(
        str(audio_path),
        fp16=False,  # Use FP32 for better CPU compatibility
    )

    return {
        'language': result.get('language', 'unknown'),
        'text': result.get('text', '').strip(),
    }


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def is_english_lyrics(
    analysis: dict,
    sample_duration_ms: int,
    min_words_per_minute: int = 30
) -> bool:
    """
    Determine if the audio sample contains English lyrics.

    Criteria:
    - Language detected as English
    - Word density meets minimum threshold (filters out sparse dialogue)

    Args:
        analysis: Dict with 'language' and 'text' from Whisper
        sample_duration_ms: Duration of the analyzed sample
        min_words_per_minute: Minimum words per minute to qualify as lyrics
                              (typical lyrics: 40-80 wpm, dialogue: 10-30 wpm)
    """
    if analysis['language'] != 'en':
        return False

    text = analysis['text']
    word_count = count_words(text)

    # Calculate words per minute
    duration_minutes = sample_duration_ms / 60000
    if duration_minutes <= 0:
        return False

    words_per_minute = word_count / duration_minutes

    return words_per_minute >= min_words_per_minute


def classify_audio(
    model,
    audio_path: Path,
    sample_duration_ms: int = 30000,
    min_words_per_minute: int = 30,
) -> tuple[str, dict]:
    """
    Classify a single audio file using multi-sample analysis.

    Extracts 3 samples from different positions (25%, 50%, 75%) and classifies
    based on how many samples pass the English lyrics test.

    Args:
        model: Loaded Whisper model
        audio_path: Path to audio file
        sample_duration_ms: Duration of each sample (default: 30s)
        min_words_per_minute: Minimum word density threshold

    Returns:
        (category, analysis_details) where category is:
        - 'songs': 2+ samples passed (confident English lyrics)
        - 'ambiguous': 1 sample passed (movie scenes, sparse vocals)
        - 'other': 0 samples passed (instrumental/non-English)
    """
    temp_files = []
    try:
        # Extract 3 samples from different positions
        samples = extract_multiple_samples(audio_path, sample_duration_ms)
        temp_files = [s[0] for s in samples]

        # Analyze each sample
        passed_samples = 0
        all_text = []
        detected_languages = []

        for temp_path, duration in samples:
            analysis = analyze_audio(model, temp_path)
            all_text.append(analysis['text'])
            detected_languages.append(analysis['language'])

            if is_english_lyrics(analysis, duration, min_words_per_minute):
                passed_samples += 1

        # Determine category based on passed samples
        if passed_samples >= 2:
            category = 'songs'
        elif passed_samples == 1:
            category = 'ambiguous'
        else:
            category = 'other'

        # Combine analysis info for reporting
        combined_analysis = {
            'language': max(set(detected_languages), key=detected_languages.count),
            'text': ' | '.join(all_text),
            'samples_passed': passed_samples,
            'samples_total': len(samples),
        }

        return category, combined_analysis

    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


def process_folder(
    input_folder: Path,
    output_songs: Path,
    output_ambiguous: Path,
    output_other: Path,
    model_name: str = "base",
    sample_duration: int = 30,
    min_words_per_minute: int = 30,
    dry_run: bool = False,
    copy_files: bool = False,
):
    """
    Process all audio files in the input folder and classify them into 3 categories.
    """
    # Create output directories
    if not dry_run:
        output_songs.mkdir(parents=True, exist_ok=True)
        output_ambiguous.mkdir(parents=True, exist_ok=True)
        output_other.mkdir(parents=True, exist_ok=True)

    # Load Whisper model
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("Model loaded.\n")

    # Find all audio files (MP3 and M4A)
    audio_files = (
        list(input_folder.glob("*.mp3")) +
        list(input_folder.glob("*.MP3")) +
        list(input_folder.glob("*.m4a")) +
        list(input_folder.glob("*.M4A"))
    )

    if not audio_files:
        print(f"No MP3 or M4A files found in {input_folder}")
        return

    print(f"Found {len(audio_files)} audio file(s) to process.\n")

    results = {
        'songs': [],
        'ambiguous': [],
        'other': [],
        'errors': [],
    }

    # Map categories to output folders
    category_folders = {
        'songs': output_songs,
        'ambiguous': output_ambiguous,
        'other': output_other,
    }

    progress_bar = tqdm(
        audio_files,
        desc="Classifying",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for audio_file in progress_bar:
        progress_bar.set_postfix_str(audio_file.name[:30])

        try:
            category, analysis = classify_audio(
                model,
                audio_file,
                sample_duration_ms=sample_duration * 1000,
                min_words_per_minute=min_words_per_minute,
            )

            dest_folder = category_folders[category]
            results[category].append(audio_file.name)

            # Move or copy file
            if not dry_run:
                dest_path = dest_folder / audio_file.name
                if copy_files:
                    shutil.copy2(audio_file, dest_path)
                else:
                    shutil.move(str(audio_file), str(dest_path))

        except Exception as e:
            results['errors'].append((audio_file.name, str(e)))

    progress_bar.close()
    print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Songs (English lyrics):  {len(results['songs'])} files")
    print(f"Ambiguous (movie/other): {len(results['ambiguous'])} files")
    print(f"Other (instrumental):    {len(results['other'])} files")
    print(f"Errors:                  {len(results['errors'])} files")

    if results['errors']:
        print("\nFiles with errors:")
        for name, error in results['errors']:
            print(f"  - {name}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify audio files (MP3/M4A) into songs, ambiguous, and other categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  songs/      - English lyrics (2+ of 3 samples pass word density check)
  ambiguous/  - Movie scenes, sparse dialogue (1 of 3 samples pass)
  other/      - Instrumental or non-English (0 samples pass)

Examples:
  %(prog)s /path/to/music
  %(prog)s /path/to/music --model small
  %(prog)s /path/to/music --dry-run
  %(prog)s /path/to/music --copy
        """
    )

    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing MP3/M4A files to classify"
    )

    parser.add_argument(
        "--output-songs",
        type=Path,
        default=None,
        help="Folder for songs with English lyrics (default: input_folder/songs)"
    )

    parser.add_argument(
        "--output-ambiguous",
        type=Path,
        default=None,
        help="Folder for ambiguous files like movie scenes (default: input_folder/ambiguous)"
    )

    parser.add_argument(
        "--output-other",
        type=Path,
        default=None,
        help="Folder for instrumental/non-English files (default: input_folder/other)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = more accurate but slower"
    )

    parser.add_argument(
        "--sample-duration",
        type=int,
        default=30,
        help="Duration in seconds of each audio sample (default: 30). Three samples are taken per file."
    )

    parser.add_argument(
        "--min-wpm",
        type=int,
        default=30,
        help="Minimum words per minute to classify as lyrics (default: 30). Typical lyrics: 40-80 wpm."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files"
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them"
    )

    args = parser.parse_args()

    # Validate input folder
    if not args.input_folder.exists():
        print(f"Error: Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    if not args.input_folder.is_dir():
        print(f"Error: Input path is not a folder: {args.input_folder}")
        sys.exit(1)

    # Set default output folders
    output_songs = args.output_songs or (args.input_folder / "songs")
    output_ambiguous = args.output_ambiguous or (args.input_folder / "ambiguous")
    output_other = args.output_other or (args.input_folder / "other")

    # Run classification
    process_folder(
        input_folder=args.input_folder,
        output_songs=output_songs,
        output_ambiguous=output_ambiguous,
        output_other=output_other,
        model_name=args.model,
        sample_duration=args.sample_duration,
        min_words_per_minute=args.min_wpm,
        dry_run=args.dry_run,
        copy_files=args.copy,
    )


if __name__ == "__main__":
    main()
