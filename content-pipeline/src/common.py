import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DRAFTS_DIR = OUTPUTS_DIR / "drafts"

SOURCE_VIDEOS_DIR = INPUTS_DIR / "source_videos"
GAMEPLAY_LOOPS_DIR = INPUTS_DIR / "gameplay_loops"
PENDING_CLIPS_DIR = INPUTS_DIR / "pending_clips"
INGEST_DIR = DATA_DIR / "ingest"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
SCRIPTS_DIR = DATA_DIR / "scripts"
RENDER_PLANS_DIR = DATA_DIR / "render_plans"
VOICE_DIR = DATA_DIR / "voice"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def load_config() -> dict:
    with open(CONFIG_DIR / "models.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_effect_presets() -> dict:
    with open(CONFIG_DIR / "effect_presets.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_ingest_record(record_id: str) -> dict:
    path = INGEST_DIR / f"{record_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No ingest record found for id '{record_id}' at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cleanup_intermediates(record_id: str) -> None:
    """Deletes every per-record intermediate produced upstream of render()
    (transcript, render plan, script, and voice files) plus the .ass caption
    file render() burns into the final video, leaving source videos and
    outputs/drafts/{record_id}.mp4 untouched. Safe to call once render() has
    succeeded - nothing removed here is read again after that point."""
    for path in (
        INGEST_DIR / f"{record_id}.json",
        TRANSCRIPTS_DIR / f"{record_id}.wav",
        TRANSCRIPTS_DIR / f"{record_id}.json",
        RENDER_PLANS_DIR / f"{record_id}.json",
        SCRIPTS_DIR / f"{record_id}.txt",
        VOICE_DIR / f"{record_id}.wav",
        VOICE_DIR / f"{record_id}.json",
        DRAFTS_DIR / f"{record_id}.ass",
    ):
        path.unlink(missing_ok=True)


def call_llm(messages: list[dict], config: dict) -> str:
    response = requests.post(
        config["endpoint"],
        json={
            "model": config["model"],
            "messages": messages,
            "format": "json",
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def flag_loud_segments(audio_path: Path, segments: list[dict], spike_ratio: float = 1.3) -> None:
    """Marks each segment dict with segment["loud"] = True if its peak audio
    energy notably exceeds the file's baseline - a cheap proxy for an actual
    excited vocal reaction (yelling, laughing, hype) versus flat/quiet dead
    air, since transcript text alone can't distinguish the two. Mutates
    segments in place; audio_path must be the mono PCM wav used to transcribe
    them (extract_audio's output)."""
    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    window_samples = int(sample_rate * 0.5)
    if samples.size == 0 or window_samples <= 0:
        return

    n_windows = max(len(samples) // window_samples, 1)
    trimmed = samples[: n_windows * window_samples]
    rms = np.sqrt(np.mean(trimmed.reshape(n_windows, window_samples) ** 2, axis=1))
    nonzero = rms[rms > 0]
    # The 85th percentile (not the median) so baseline reflects the streamer's
    # normal loud-talking level, not a value dragged down by quiet gaps - a
    # constant talker's median is close to silence-adjacent, which would flag
    # nearly all ordinary speech as "loud" and make the signal meaningless.
    baseline = float(np.percentile(nonzero, 85)) if nonzero.size else 0.0
    if baseline <= 0:
        return

    for seg in segments:
        start_w = max(int(seg["start"] / 0.5), 0)
        end_w = min(int(seg["end"] / 0.5) + 1, n_windows)
        if start_w >= end_w:
            continue
        peak = float(np.max(rms[start_w:end_w]))
        seg["loud"] = peak >= baseline * spike_ratio


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr}")
    return float(result.stdout.strip())
