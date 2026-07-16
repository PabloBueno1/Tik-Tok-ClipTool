import argparse
import sys
import traceback
from pathlib import Path

from common import VIDEO_EXTENSIONS, cleanup_intermediates
from generate_voice import generate_voice
from ingest import ingest
from plan_video import plan_and_save, plan_raw_and_save
from render import render, render_raw
from split_movie import (
    DEFAULT_MAX_CLIP_SECONDS,
    DEFAULT_MAX_CLIPS_PER_WINDOW,
    DEFAULT_MAX_TOTAL_CLIPS,
    DEFAULT_MIN_CLIP_SECONDS,
    DEFAULT_MIN_SCORE,
    DEFAULT_WINDOW_SECONDS,
    split_into_highlights,
)
from transcribe import transcribe


def run_pipeline(video_path: Path, title: str, source: str, raw: bool = False) -> Path:
    record_id = ingest(video_path, title, source)
    print(f"[1/5] Ingested: {record_id}")

    transcribe(record_id)
    print("[2/5] Transcribed")

    if raw:
        plan_raw_and_save(record_id)
        print("[3/5] Raw plan generated (hook + teaser only)")
        print("[4/5] Voiceover skipped (raw mode - streamer audio is the audio)")
        final_path = render_raw(record_id)
    else:
        plan_and_save(record_id)
        print("[3/5] Render plan generated")

        generate_voice(record_id)
        print("[4/5] Voiceover generated")

        final_path = render(record_id)
    print(f"[5/5] Rendered: {final_path}")

    cleanup_intermediates(record_id)
    print("Cleaned up intermediate files")

    return final_path


def run_many(video_paths: list[Path], title: str, source: str, raw: bool = False) -> tuple[list[Path], list[tuple[Path, Exception]]]:
    """Runs the full pipeline for each video in video_paths. A failure on one
    (e.g. a stretch with no real dialogue that exhausts plan_video.py's
    retries) is logged and skipped rather than aborting the rest of the batch."""
    outputs = []
    failures = []
    for i, video_path in enumerate(video_paths, start=1):
        print(f"--- {i}/{len(video_paths)}: {video_path.name} ---")
        try:
            outputs.append(run_pipeline(video_path, f"{title} - {video_path.stem}", source, raw=raw))
        except Exception as e:
            print(f"FAILED on {video_path.name}: {e}", file=sys.stderr)
            traceback.print_exc()
            failures.append((video_path, e))

    print(f"\nBatch done: {len(outputs)} succeeded, {len(failures)} failed.")
    for video_path, e in failures:
        print(f"  - {video_path.name}: {e}")

    return outputs, failures


def run_batch(dir_path: Path, title: str, source: str, raw: bool = False) -> tuple[list[Path], list[tuple[Path, Exception]]]:
    """Runs the full pipeline for every video file already staged in a
    directory - the intended use is pointing this at PENDING_CLIPS_DIR."""
    video_files = sorted(p for p in dir_path.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not video_files:
        raise RuntimeError(f"No video files found in {dir_path}")
    return run_many(video_files, title, source, raw=raw)


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline end to end: (split) -> ingest -> transcribe -> plan -> voice -> render.")
    parser.add_argument("--video", required=True, help="Path to a source video file, or a directory of already-split video files to process one by one.")
    parser.add_argument("--title", required=True, help="Title/description of the clip (used as a prefix when processing multiple clips).")
    parser.add_argument("--source", default="unknown", help="Where the clip came from (e.g. a VOD URL or note).")
    parser.add_argument("--no-split", action="store_true", help="When --video is a single long-form file, skip highlight splitting and run it through the pipeline as one clip.")
    parser.add_argument("--raw", action="store_true", help="Raw mode: no TTS narration or b-roll - full-frame clip with the streamer's own audio, source-word karaoke captions, hook overlay, and cold-open teaser.")
    parser.add_argument("--window-minutes", type=float, default=DEFAULT_WINDOW_SECONDS / 60, help="Splitting: transcript window size fed to the LLM per call (default: 15).")
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_CLIP_SECONDS, help="Splitting: minimum clip length in seconds (default: 60).")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS, help="Splitting: maximum clip length in seconds (default: 90). A single --video file at or under this length is never split.")
    parser.add_argument("--max-clips-per-window", type=int, default=DEFAULT_MAX_CLIPS_PER_WINDOW, help="Splitting: max highlight candidates the LLM may pick per window (default: 4).")
    parser.add_argument("--max-total-clips", type=int, default=DEFAULT_MAX_TOTAL_CLIPS, help="Splitting: max clips kept for the whole video after the global rerank pass (default: 12).")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Splitting: candidates scoring below this (1-10) are dropped before reranking (default: 5).")
    parser.add_argument("--chat", default=None, help="Splitting: path to a chat replay file (yt-dlp .live_chat.json or chat-downloader JSON). Default: auto-detect <video>.live_chat.json next to the video.")
    args = parser.parse_args()

    path = Path(args.video)
    if path.is_dir():
        outputs, _ = run_batch(path, args.title, args.source, raw=args.raw)
        for output_path in outputs:
            print(output_path)
    elif args.no_split:
        print(run_pipeline(path, args.title, args.source, raw=args.raw))
    else:
        clip_paths = split_into_highlights(
            path,
            window_seconds=args.window_minutes * 60,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            max_clips_per_window=args.max_clips_per_window,
            max_total_clips=args.max_total_clips,
            min_score=args.min_score,
            context=args.title,
            chat_path=Path(args.chat) if args.chat else None,
        )
        if clip_paths == [path]:
            print(run_pipeline(path, args.title, args.source, raw=args.raw))
        else:
            outputs, _ = run_many(clip_paths, args.title, args.source, raw=args.raw)
            for output_path in outputs:
                print(output_path)


if __name__ == "__main__":
    main()
