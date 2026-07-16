import argparse
import json
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common import CONFIG_DIR, SCRIPTS_DIR, VOICE_DIR, load_config

PIPER_VOICES_DIR = CONFIG_DIR / "piper_voices"
EDGE_FALLBACK_VOICE = "en-US-GuyNeural"


def tool_path(name: str) -> str:
    """Resolves a CLI tool installed into this venv's Scripts/bin directory
    (piper, edge-tts) so the pipeline works even when the venv isn't
    activated; falls back to bare name -> PATH lookup."""
    for candidate in (Path(sys.executable).parent / f"{name}.exe", Path(sys.executable).parent / name):
        if candidate.exists():
            return str(candidate)
    return name


def resolve_piper_voice(voice: str) -> str:
    local_model = PIPER_VOICES_DIR / f"{voice}.onnx"
    return str(local_model) if local_model.exists() else voice


def load_script(record_id: str) -> str:
    path = SCRIPTS_DIR / f"{record_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No script found for id '{record_id}' at {path}")
    return path.read_text(encoding="utf-8").strip()


def synthesize_piper(text: str, voice: str, out_path: Path) -> None:
    cmd = [tool_path("piper"), "--model", voice, "--output_file", str(out_path)]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"piper TTS failed:\n{result.stderr}")


def synthesize_edge_tts(text: str, voice: str, out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        text_path = Path(f.name)

    try:
        cmd = [tool_path("edge-tts"), "--file", str(text_path), "--voice", voice, "--write-media", str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"edge-tts failed:\n{result.stderr}")
    finally:
        text_path.unlink(missing_ok=True)


def synthesize(text: str, config: dict, out_path: Path) -> tuple[str, str]:
    provider = config["provider"]
    fallback = config.get("fallback_provider")
    piper_voice = random.choice(config["voices"]) if "voices" in config else config.get("voice")

    try:
        if provider == "piper":
            synthesize_piper(text, resolve_piper_voice(piper_voice), out_path)
            return provider, piper_voice
        elif provider == "edge-tts":
            voice = config.get("fallback_voice", EDGE_FALLBACK_VOICE)
            synthesize_edge_tts(text, voice, out_path)
            return provider, voice
        else:
            raise RuntimeError(f"Unknown TTS provider: {provider}")
    except (RuntimeError, OSError) as e:
        if not fallback or fallback == provider or fallback != "edge-tts":
            raise
        print(f"Primary TTS provider '{provider}' failed ({e}); falling back to '{fallback}'", file=sys.stderr)
        voice = config.get("fallback_voice", EDGE_FALLBACK_VOICE)
        synthesize_edge_tts(text, voice, out_path)
        return fallback, voice


def align_audio(audio_path: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    config = load_config()["transcription"]
    model = WhisperModel(
        config["model"],
        device=config["device"],
        compute_type="float16" if config["device"] == "cuda" else "int8",
    )

    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            })
    return words


def generate_voice(record_id: str) -> Path:
    script_text = load_script(record_id)
    config = load_config()["tts"]

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = VOICE_DIR / f"{record_id}.wav"

    provider_used, voice_used = synthesize(script_text, config, audio_path)
    words = align_audio(audio_path)

    alignment = {
        "id": record_id,
        "audio_path": str(audio_path),
        "tts_provider": provider_used,
        "tts_voice": voice_used,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "words": words,
    }

    alignment_path = VOICE_DIR / f"{record_id}.json"
    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(alignment, f, indent=2)

    return audio_path


def main():
    parser = argparse.ArgumentParser(description="Generate voiceover audio and word-level alignment from a render plan script.")
    parser.add_argument("--id", required=True, help="Render plan id from plan_video.py.")
    args = parser.parse_args()

    audio_path = generate_voice(args.id)
    print(audio_path)


if __name__ == "__main__":
    main()
