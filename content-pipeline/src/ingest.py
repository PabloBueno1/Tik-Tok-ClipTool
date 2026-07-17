import argparse
import json
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from common import INGEST_DIR, SOURCE_VIDEOS_DIR
from detect_cam import detect_cam_region


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "clip"


def ingest(video_path: Path, title: str, source: str, teaser: dict | None = None) -> str:
    """teaser: optional {"start", "end"} in clip-local seconds - the measured
    peak moment chosen by split_movie's moment re-centering, carried through
    so the plan stage never has to (unreliably) rediscover it."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    record_id = f"{slugify(title)}_{uuid.uuid4().hex[:8]}"
    dest_path = SOURCE_VIDEOS_DIR / f"{record_id}{video_path.suffix}"

    SOURCE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    INGEST_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(video_path, dest_path)

    record = {
        "id": record_id,
        "title": title,
        "source": source,
        "original_path": str(video_path.resolve()),
        "video_path": str(dest_path.resolve()),
        "cam_region": detect_cam_region(dest_path),
        "teaser": teaser,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    record_path = INGEST_DIR / f"{record_id}.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record_id


def main():
    parser = argparse.ArgumentParser(description="Ingest a source video into the pipeline.")
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    parser.add_argument("--title", required=True, help="Title/description of the clip.")
    parser.add_argument("--source", default="unknown", help="Where the clip came from (e.g. a VOD URL or note).")
    args = parser.parse_args()

    record_id = ingest(Path(args.video), args.title, args.source)
    print(record_id)


if __name__ == "__main__":
    main()
