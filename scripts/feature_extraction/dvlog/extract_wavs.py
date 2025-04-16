import os
import sys
import logging
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def verify_ffmpeg():
    """Verify that ffmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def convert_video_to_wav(video_path, wav_path):
    """Convert a video file to WAV format using ffmpeg."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i", video_path,
                "-ar", "16000",  # Set audio sampling rate to 16kHz
                "-ac", "1",      # Convert to mono
                "-acodec", "pcm_s16le",  # Use PCM 16-bit encoding
                wav_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                result.stdout,
                result.stderr
            )
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {video_path}: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Script to extract WAV audio from video files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--csv-path",
        default="./data/D-vlog/video_ids.csv",
        type=str,
        help="CSV file containing video IDs."
    )
    parser.add_argument(
        "--column-video-id",
        default="video_id",
        type=str,
        help="Name of the CSV column containing video IDs."
    )
    parser.add_argument(
        "--video-dir",
        default="./data/D-vlog/videos/",
        type=str,
        help="Directory containing the video files."
    )
    parser.add_argument(
        "--dest-dir",
        default="./data/D-vlog/wavs/",
        type=str,
        help="Directory where WAV files will be stored."
    )

    args = parser.parse_args()
    setup_logging()

    # Check if ffmpeg is installed
    if not verify_ffmpeg():
        logging.error("ffmpeg is not installed. Please install it first.")
        sys.exit(1)

    # Verify input CSV file exists
    if not os.path.exists(args.csv_path):
        logging.error(f"CSV file not found: {args.csv_path}")
        sys.exit(1)

    # Verify video directory exists
    if not os.path.exists(args.video_dir):
        logging.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.dest_dir, exist_ok=True)
    logging.info(f"Output directory: {args.dest_dir}")

    try:
        # Read CSV and convert video IDs to strings
        df = pd.read_csv(args.csv_path)
        if args.column_video_id not in df.columns:
            logging.error(f"Column '{args.column_video_id}' not found in CSV file")
            sys.exit(1)
            
        video_ids = df[args.column_video_id].astype(str).tolist()
        logging.info(f"Found {len(video_ids)} video IDs in CSV file")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Process each video
    successful = 0
    failed = 0
    skipped = 0

    for video_id in tqdm(video_ids, desc="Converting videos to WAV"):
        wav_path = os.path.join(args.dest_dir, f"{video_id}.wav")
        video_path = os.path.join(args.video_dir, f"{video_id}.mp4")

        # Skip if WAV file already exists
        if os.path.exists(wav_path):
            logging.debug(f"Skipping {video_id}: WAV file already exists")
            skipped += 1
            continue

        # Convert video to WAV
        if convert_video_to_wav(video_path, wav_path):
            successful += 1
        else:
            failed += 1

    # Print summary
    logging.info(f"""
Conversion complete:
- Successful: {successful}
- Failed: {failed}
- Skipped: {skipped}
- Total processed: {len(video_ids)}
""")

if __name__ == "__main__":
    main()
