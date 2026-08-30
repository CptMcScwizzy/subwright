#!/usr/bin/env python3
"""
Video Translation Watcher
Monitors an ingest folder, translates videos to English subtitles using GPU-accelerated Whisper,
and organizes output into folders.
Features:
- Drop files in 'ingest' folder to process and organize into subfolders
- Drop files in 'reprocess' folder to regenerate subtitles in place
- GPU-accelerated transcription with CUDA
- Progress logging during transcription
Usage:
    python translate_watcher.py [--watch-dir /path/to/translate] [--model large-v3]
"""
import os
import sys
import time
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
log_handlers = [logging.StreamHandler()]
log_file = Path.home() / "translate_watcher.log"
try:
    log_handlers.append(logging.FileHandler(log_file))
except:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
'.m4v', '.ts'}

# Global model (loaded once)
model = None

def load_model(model_size: str):
    """Load the Whisper model with GPU acceleration."""
    global model
    if model is not None:
        return model

    from faster_whisper import WhisperModel

    logger.info(f"Loading {model_size} model with CUDA...")

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8")
        logger.info("Model loaded successfully on GPU")
    except Exception as e:
        logger.warning(f"GPU loading failed ({e}), falling back to CPU with INT8")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Model loaded on CPU")

    return model

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def remux_if_needed(video_path: Path) -> Path:
    """Check if ffprobe can find an audio stream. If not, try re-muxing as mpegts to fix bad headers."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            return video_path  # Audio stream found, file is fine
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return video_path  # ffprobe not available or timed out, skip check

    # No audio stream found — try re-muxing as mpegts
    logger.warning(f"No audio stream detected in {video_path.name}, attempting remux as mpegts...")
    fixed_path = video_path.with_name(f"{video_path.stem}_remuxed{video_path.suffix}")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", str(video_path),
             "-c", "copy", str(fixed_path)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0 and fixed_path.exists() and fixed_path.stat().st_size > 0:
            # Verify the remuxed file has audio
            verify = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(fixed_path)],
                capture_output=True, text=True, timeout=30
            )
            if verify.stdout.strip():
                # Replace original with remuxed version
                video_path.unlink()
                fixed_path.rename(video_path)
                logger.info(f"Successfully remuxed {video_path.name} — audio stream now accessible")
                return video_path
            else:
                fixed_path.unlink()
                logger.warning(f"Remuxed file still has no audio, using original")
        else:
            if fixed_path.exists():
                fixed_path.unlink()
            logger.warning(f"Remux failed: {result.stderr[:200]}")
    except (subprocess.TimeoutExpired, Exception) as e:
        if fixed_path.exists():
            fixed_path.unlink()
        logger.warning(f"Remux error: {e}")

    return video_path

def generate_subtitles(video_path: Path, output_srt: Path, model_size: str, source_language: str = None):
    """Generate translated English subtitles for a video file."""
    # Fix files with bad headers (e.g., PNG header prepended to video data)
    video_path = remux_if_needed(video_path)

    whisper_model = load_model(model_size)

    logger.info(f"Transcribing: {video_path.name}")
    start_time = time.time()

    # Transcribe and translate with tuned settings for better dialogue capture
    transcribe_opts = {
        "task": "translate",
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 300,      # Shorter silence detection
            "speech_pad_ms": 200,                # Pad speech segments
            "threshold": 0.3,                    # Lower = catches quieter speech
        },
        "no_speech_threshold": 0.4,              # Lower = less likely to skip quiet parts
        "compression_ratio_threshold": 2.8,      # Higher = more lenient
        "condition_on_previous_text": True,      # Better context continuity
    }

    if source_language:
        transcribe_opts["language"] = source_language

    segments, info = whisper_model.transcribe(str(video_path), **transcribe_opts)

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    logger.info(f"Processing audio with duration {format_duration(info.duration)}")

    # Max subtitle duration (seconds)
    MAX_SUBTITLE_DURATION = 5.0

    # Write SRT file with progress logging
    segment_count = 0
    last_log_time = time.time()

    with open(output_srt, "w", encoding="utf-8") as srt_file:
        for segment in segments:
            segment_count += 1
            i = segment_count

            try:
                seg_start = segment.start
                seg_end = segment.end
                text = segment.text.strip()
            except (IndexError, AttributeError) as e:
                logger.warning(f"Skipping malformed segment {i}: {e}")
                continue

            if not text:
                continue

            # Cap subtitle duration
            duration = seg_end - seg_start
            end_time = seg_start + min(duration, MAX_SUBTITLE_DURATION)

            start = format_timestamp(seg_start)
            end = format_timestamp(end_time)

            srt_file.write(f"{i}\n")
            srt_file.write(f"{start} --> {end}\n")
            srt_file.write(f"{text}\n\n")

            # Log progress every 30 seconds or every 50 segments
            current_time = time.time()
            if current_time - last_log_time >= 30 or i % 50 == 0:
                progress_pct = (seg_end / info.duration) * 100 if info.duration > 0 else 0
                elapsed = current_time - start_time
                eta = (elapsed / progress_pct) * (100 - progress_pct) if progress_pct > 0 else 0
                logger.info(
                    f"Progress: {progress_pct:.1f}% | "
                    f"Segment {i} @ {format_timestamp(seg_start)} | "
                    f"Elapsed: {format_duration(elapsed)} | "
                    f"ETA: {format_duration(eta)}"
                )
                last_log_time = current_time

    elapsed = time.time() - start_time
    if segment_count == 0:
        logger.warning(f"No speech segments found in {video_path.name}")
    logger.info(f"Completed: {segment_count} segments in {format_duration(elapsed)} -> {output_srt.name}")

def process_video(video_path: Path, base_dir: Path, model_size: str, source_language: str = None):
    """Process a single video file: create folder, move file, generate subtitles."""
    # Create output folder named after the video (without extension)
    folder_name = video_path.stem
    output_folder = base_dir / folder_name

    # Handle duplicate folder names
    if output_folder.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{video_path.stem}_{timestamp}"
        output_folder = base_dir / folder_name

    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created folder: {output_folder}")

    # Define paths
    new_video_path = output_folder / video_path.name
    srt_path = output_folder / f"{video_path.stem}.srt"

    # Move video to output folder
    shutil.move(str(video_path), str(new_video_path))
    logger.info(f"Moved video to: {new_video_path}")

    # Generate subtitles
    try:
        generate_subtitles(new_video_path, srt_path, model_size, source_language)
        # Create a marker file to indicate success
        marker = output_folder / ".translated"
        marker.write_text(f"Translated on {datetime.now().isoformat()}\n")
    except Exception as e:
        logger.error(f"Failed to generate subtitles: {e}")
        # Create error marker
        error_marker = output_folder / ".translation_error"
        error_marker.write_text(f"Error: {e}\nTime: {datetime.now().isoformat()}\n")
        raise

def reprocess_video(video_path: Path, model_size: str, source_language: str = None):
    """Reprocess a video file: regenerate subtitles in place without moving."""
    srt_path = video_path.with_suffix('.srt')

    logger.info(f"Reprocessing: {video_path.name}")

    # Backup existing srt if it exists
    if srt_path.exists():
        backup_path = srt_path.with_suffix('.srt.bak')
        shutil.copy(str(srt_path), str(backup_path))
        logger.info(f"Backed up existing subtitles to: {backup_path.name}")

    try:
        generate_subtitles(video_path, srt_path, model_size, source_language)
        # Remove from reprocess folder by creating marker
        marker = video_path.parent / f".reprocessed_{video_path.stem}"
        marker.write_text(f"Reprocessed on {datetime.now().isoformat()}\n")
    except Exception as e:
        logger.error(f"Failed to reprocess: {e}")
        # Restore backup if it exists
        backup_path = srt_path.with_suffix('.srt.bak')
        if backup_path.exists():
            shutil.move(str(backup_path), str(srt_path))
            logger.info("Restored backup subtitles")
        raise

def get_pending_videos(ingest_dir: Path) -> list:
    """Get list of video files ready for processing."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(ingest_dir.glob(f"*{ext}"))
        videos.extend(ingest_dir.glob(f"*{ext.upper()}"))

    # Filter out files that are still being written (modified in last 10 seconds)
    ready_videos = []
    for video in videos:
        try:
            mtime = video.stat().st_mtime
            if time.time() - mtime > 10:
                ready_videos.append(video)
            else:
                logger.debug(f"Skipping {video.name} - still being written")
        except OSError:
            continue

    return sorted(ready_videos, key=lambda p: p.stat().st_mtime)

def get_reprocess_videos(reprocess_dir: Path) -> list:
    """Get list of video files to reprocess."""
    if not reprocess_dir.exists():
        return []

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(reprocess_dir.glob(f"*{ext}"))
        videos.extend(reprocess_dir.glob(f"*{ext.upper()}"))

    # Filter out files that are still being written
    ready_videos = []
    for video in videos:
        try:
            mtime = video.stat().st_mtime
            marker = video.parent / f".reprocessed_{video.stem}"
            # Skip if already reprocessed or still being written
            if time.time() - mtime > 10 and not marker.exists():
                ready_videos.append(video)
        except OSError:
            continue

    return sorted(ready_videos, key=lambda p: p.stat().st_mtime)

def watch_loop(base_dir: Path, model_size: str, source_language: str = None,
poll_interval: int = 30):
    """Main watch loop - monitor ingest folder and process videos."""
    ingest_dir = base_dir / "ingest"
    reprocess_dir = base_dir / "reprocess"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    reprocess_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Watching for new videos in: {ingest_dir}")
    logger.info(f"Watching for reprocess videos in: {reprocess_dir}")
    logger.info(f"Output folders will be created in: {base_dir}")
    logger.info(f"Model: {model_size}, Source language: {source_language or 'auto-detect'}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info("=" * 60)

    # Pre-load model
    load_model(model_size)

    while True:
        try:
            # Check for new videos in ingest folder
            videos = get_pending_videos(ingest_dir)
            for video in videos:
                logger.info(f"Found new video: {video.name}")
                try:
                    process_video(video, base_dir, model_size, source_language)
                except Exception as e:
                    logger.error(f"Error processing {video.name}: {e}")
                    continue

            # Check for videos to reprocess
            reprocess_videos = get_reprocess_videos(reprocess_dir)
            for video in reprocess_videos:
                logger.info(f"Found video to reprocess: {video.name}")
                try:
                    reprocess_video(video, model_size, source_language)
                except Exception as e:
                    logger.error(f"Error reprocessing {video.name}: {e}")
                    continue

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Watch loop error: {e}")
            time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(description="Watch folder and auto-translate videos to English subtitles")
    parser.add_argument(
        "--watch-dir", "-w",
        type=str,
        default="/mnt/data/translate",
        help="Base directory (will create 'ingest' and 'reprocess' subfolders)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Source language code (e.g., 'ja' for Japanese). Auto-detect if not specified."
    )
    parser.add_argument(
        "--poll-interval", "-p",
        type=int,
        default=30,
        help="Seconds between folder checks (default: 30)"
    )

    args = parser.parse_args()
    base_dir = Path(args.watch_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    watch_loop(base_dir, args.model, args.language, args.poll_interval)

if __name__ == "__main__":
    main()
