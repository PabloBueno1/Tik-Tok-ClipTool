import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel

from common import TRANSCRIPTS_DIR, load_config, load_ingest_record


def extract_audio(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def transcribe_segments(audio_path: Path) -> tuple[list[dict], object]:
    """Runs faster-whisper over an audio file and returns (segment_list, info).
    Standalone from any ingest record, so it's reusable for pre-ingest passes
    (e.g. split_movie.py transcribing a raw long-form download)."""
    config = load_config()["transcription"]
    model = WhisperModel(
        config["model"],
        device=config["device"],
        compute_type="float16" if config["device"] == "cuda" else "int8",
    )

    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    segment_list = []
    for seg in segments:
        segment_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [
                {"word": w.word.strip(), "start": w.start, "end": w.end, "probability": w.probability}
                for w in (seg.words or [])
            ],
        })

    return segment_list, info


def transcribe(record_id: str) -> Path:
    record = load_ingest_record(record_id)
    video_path = Path(record["video_path"])

    audio_path = TRANSCRIPTS_DIR / f"{record_id}.wav"
    extract_audio(video_path, audio_path)

    segment_list, info = transcribe_segments(audio_path)

    transcript = {
        "id": record_id,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "language": info.language,
        "duration": info.duration,
        "model": load_config()["transcription"]["model"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segments": segment_list,
    }

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPTS_DIR / f"{record_id}.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)

    return transcript_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe an ingested source video.")
    parser.add_argument("--id", required=True, help="Ingest record id from ingest.py.")
    args = parser.parse_args()

    transcript_path = transcribe(args.id)
    print(transcript_path)


if __name__ == "__main__":
    main()
