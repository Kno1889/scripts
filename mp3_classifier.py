#!/usr/bin/env python3
"""
MP3 File Classifier
Classifies MP3 files into English lyrics vs non-English/instrumental
using audio analysis with OpenAI's Whisper model.
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


def extract_audio_sample(mp3_path: Path, sample_duration_ms: int = 60000) -> Path:
    """
    Extract a sample from the middle of the audio file for analysis.
    Analyzing the middle helps avoid intros/outros that may not have vocals.
    """
    audio = AudioSegment.from_mp3(mp3_path)

    # Get sample from the middle of the track
    duration = len(audio)
    if duration <= sample_duration_ms:
        sample = audio
    else:
        start = (duration - sample_duration_ms) // 2
        sample = audio[start:start + sample_duration_ms]

    # Export as WAV for Whisper (better compatibility)
    temp_path = mp3_path.with_suffix('.temp.wav')
    sample.export(temp_path, format='wav')
    return temp_path


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


def is_english_lyrics(analysis: dict, min_text_length: int = 20) -> bool:
    """
    Determine if the audio contains English lyrics.

    Criteria:
    - Language detected as English
    - Has substantial transcribed text (not just noise/silence)
    """
    if analysis['language'] != 'en':
        return False

    # Check if there's enough text to consider it "lyrics"
    text = analysis['text']
    if len(text) < min_text_length:
        return False

    return True


def classify_mp3(model, mp3_path: Path, sample_duration_ms: int = 60000) -> tuple[bool, dict]:
    """
    Classify a single MP3 file.
    Returns (is_english, analysis_details).
    """
    temp_wav = None
    try:
        # Extract sample for analysis
        temp_wav = extract_audio_sample(mp3_path, sample_duration_ms)

        # Analyze with Whisper
        analysis = analyze_audio(model, temp_wav)

        # Determine classification
        has_english = is_english_lyrics(analysis)

        return has_english, analysis

    finally:
        # Clean up temp file
        if temp_wav and temp_wav.exists():
            temp_wav.unlink()


def process_folder(
    input_folder: Path,
    output_english: Path,
    output_other: Path,
    model_name: str = "base",
    sample_duration: int = 60,
    dry_run: bool = False,
    copy_files: bool = False,
):
    """
    Process all MP3 files in the input folder and classify them.
    """
    # Create output directories
    if not dry_run:
        output_english.mkdir(parents=True, exist_ok=True)
        output_other.mkdir(parents=True, exist_ok=True)

    # Load Whisper model
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("Model loaded.\n")

    # Find all MP3 files
    mp3_files = list(input_folder.glob("*.mp3")) + list(input_folder.glob("*.MP3"))

    if not mp3_files:
        print(f"No MP3 files found in {input_folder}")
        return

    print(f"Found {len(mp3_files)} MP3 file(s) to process.\n")

    results = {
        'english': [],
        'other': [],
        'errors': [],
    }

    for i, mp3_file in enumerate(mp3_files, 1):
        print(f"[{i}/{len(mp3_files)}] Processing: {mp3_file.name}")

        try:
            is_english, analysis = classify_mp3(
                model,
                mp3_file,
                sample_duration_ms=sample_duration * 1000
            )

            lang = analysis['language']
            text_preview = analysis['text'][:80] + "..." if len(analysis['text']) > 80 else analysis['text']

            if is_english:
                print(f"  → English lyrics detected (lang={lang})")
                print(f"    Preview: \"{text_preview}\"")
                dest_folder = output_english
                results['english'].append(mp3_file.name)
            else:
                reason = "no English" if lang != 'en' else "insufficient text"
                print(f"  → Non-English/Instrumental ({reason}, lang={lang})")
                if text_preview:
                    print(f"    Preview: \"{text_preview}\"")
                dest_folder = output_other
                results['other'].append(mp3_file.name)

            # Move or copy file
            if not dry_run:
                dest_path = dest_folder / mp3_file.name
                if copy_files:
                    shutil.copy2(mp3_file, dest_path)
                    print(f"  → Copied to: {dest_folder.name}/")
                else:
                    shutil.move(str(mp3_file), str(dest_path))
                    print(f"  → Moved to: {dest_folder.name}/")
            else:
                action = "copy" if copy_files else "move"
                print(f"  → Would {action} to: {dest_folder.name}/ (dry run)")

        except Exception as e:
            print(f"  → Error: {e}")
            results['errors'].append((mp3_file.name, str(e)))

        print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"English lyrics:     {len(results['english'])} files")
    print(f"Non-English/Other:  {len(results['other'])} files")
    print(f"Errors:             {len(results['errors'])} files")

    if results['errors']:
        print("\nFiles with errors:")
        for name, error in results['errors']:
            print(f"  - {name}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify MP3 files into English lyrics vs non-English/instrumental",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/music
  %(prog)s /path/to/music --model small --sample-duration 90
  %(prog)s /path/to/music --dry-run
  %(prog)s /path/to/music --copy
        """
    )

    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing MP3 files to classify"
    )

    parser.add_argument(
        "--output-english",
        type=Path,
        default=None,
        help="Folder for English lyrics files (default: input_folder/english)"
    )

    parser.add_argument(
        "--output-other",
        type=Path,
        default=None,
        help="Folder for non-English/instrumental files (default: input_folder/other)"
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
        default=60,
        help="Duration in seconds of audio sample to analyze (default: 60)"
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
    output_english = args.output_english or (args.input_folder / "english")
    output_other = args.output_other or (args.input_folder / "other")

    # Run classification
    process_folder(
        input_folder=args.input_folder,
        output_english=output_english,
        output_other=output_other,
        model_name=args.model,
        sample_duration=args.sample_duration,
        dry_run=args.dry_run,
        copy_files=args.copy,
    )


if __name__ == "__main__":
    main()
