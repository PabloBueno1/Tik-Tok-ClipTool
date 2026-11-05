"""Downloads the Piper TTS voice models listed in config/models.json into
config/piper_voices/. The models are too large to keep in git (the
libritts-high voice alone is ~137 MB), so run this once after cloning.
Voices already on disk are skipped."""
import sys

import requests

from common import CONFIG_DIR, load_config

PIPER_VOICES_DIR = CONFIG_DIR / "piper_voices"
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_url(voice: str, suffix: str) -> str:
    # en_US-libritts-high -> en/en_US/libritts/high/en_US-libritts-high.onnx
    lang_code, name, quality = voice.split("-", 2)
    family = lang_code.split("_")[0]
    return f"{BASE_URL}/{family}/{lang_code}/{name}/{quality}/{voice}{suffix}"


def download(url: str, dest) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    tts = load_config().get("tts", {})
    voices = tts.get("voices") or ([tts["voice"]] if tts.get("voice") else [])
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        for suffix in (".onnx", ".onnx.json"):
            dest = PIPER_VOICES_DIR / f"{voice}{suffix}"
            if dest.exists():
                print(f"  have  {dest.name}")
                continue
            print(f"  fetch {dest.name}")
            download(voice_url(voice, suffix), dest)
    print(f"Voices ready in {PIPER_VOICES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
