# Content Pipeline

## Goal
Local pipeline that takes source video + a b-roll library, produces a vertical short with an LLM-authored script, voiceover, captions, and editing instructions. Posting is manual for now.

## 1. Folder Structure
```
content-pipeline/
  inputs/
    source_videos/
    gameplay_loops/
  outputs/
    drafts/
    approved/
    rejected/
  data/
    transcripts/
    scripts/
    render_plans/
  config/
    models.json
    caption_presets.json
    effect_presets.json
  src/
    ingest.py
    transcribe.py
    plan_video.py
    generate_voice.py
    render.py
    review_queue.py
```

Manual step: move an approved video from `outputs/approved/` and post it yourself.

## 2. Pipeline Stages
- **Ingest** — accept uploaded video + metadata (title, source).
- **Extract audio** — ffmpeg.
- **Transcribe** — Whisper, save transcript with word-level timestamps as JSON (source video, used for clip selection).
- **LLM Render Plan** — hook, angle, script, clip selection, captions, editing instructions.
- **Voiceover** — TTS from script, plus a word-level alignment pass on the generated audio (see §3 notes) for caption timing.
- **Render** — assemble b-roll base (randomly selected gameplay clips filling gaps), inserted source clips, captions, title card, effects/transitions/music cues.
- **Export** — vertical 1080x1920.
- **Review queue** — you look at it and approve/reject.
- **Post manually.**

## 3. Render Plan Schema (LLM Editing Instructions)
The LLM outputs structured parameters, never code — ffmpeg/moviepy stays deterministic and consumes this JSON.

```json
{
  "title": "Why this scene works",
  "hook": "This scene gives away the twist before you notice it.",
  "script": "...",
  "pacing_profile": "fast_hook_slow_body",
  "clips": [
    {
      "source_start": 12.4,
      "source_end": 16.8,
      "reason": "Shows the key reaction",
      "transition_in": "hard_cut",
      "zoom": { "start_scale": 1.0, "end_scale": 1.15 }
    }
  ],
  "graphics": [
    { "time": 3, "text": "Watch his eyes", "style": "punch_in" }
  ],
  "emphasis_words": [
    { "word": "twist", "time": 5.2, "effect": "scale_pop" }
  ],
  "music_cues": [
    { "time": 4.2, "type": "whoosh" },
    { "time": 12.0, "type": "beat_drop" }
  ],
  "effects": [
    { "time": 5.2, "type": "speed_ramp", "factor": 0.4, "duration": 0.8 },
    { "time": 8.0, "type": "color_grade", "preset": "cool_tense" },
    { "time": 12.0, "type": "screen_shake", "intensity": "medium" },
    { "time": 14.5, "type": "crop_reframe", "target": "subject_center" }
  ],
  "caption_style": {
    "preset": "bold_yellow_pop",
    "position": "center",
    "karaoke": true
  },
  "title_card": {
    "text": "This scene gives away the twist",
    "duration": 1.8,
    "animation": "fade_scale_in"
  }
}
```

Notes:
- `effects`, `caption_style`, `title_card` follow the same rule as everything else: the LLM picks which preset/timestamp, never raw filter math. Presets live in `config/effect_presets.json` / `.ass` style templates, so output stays deterministic and debuggable.
- **Captions**: text comes from the LLM script, but timing comes from a word-level forced-alignment pass over the *generated voiceover audio* (not the source video transcript — the source transcript is only for picking which source clips to use). `generate_voice.py` runs faster-whisper (or an aligner) on its own TTS output and saves word-level timestamps alongside the audio; that's what drives the `.ass` karaoke rendering. `emphasis_words` drives which words get punched up as they're spoken.
- **B-roll**: no tagging/retrieval system — `render.py` randomly selects clips (or random offsets into longer loops) from `inputs/gameplay_loops/` to fill any gap in the output timeline that isn't covered by a source clip insert. `clips` entries are placed sequentially in the order given; b-roll auto-fills the space between/around them up to the total voiceover length. No explicit output-timestamp field needed unless a future need arises to pin a clip to an exact beat.
- **Title card**: a held intro card (hook text, optional dimmed/blurred background) before the main cut, parameterized by `hook`.
- All editing instruction fields are optional-with-defaults so the render step never breaks if the LLM omits one.

## 5. Tech Stack
- Python orchestration
- ffmpeg / moviepy for video
- Local Whisper for transcription
- LLM (local Qwen via Ollama, or cloud API) for script + render plan + editing instructions
- TTS: local (Piper/XTTS-v2) or cloud (edge-tts/ElevenLabs/OpenAI)
- `.ass`/`.srt` burned in via ffmpeg
- Streamlit/FastAPI for the review dashboard

### 5a. Local-First Model Stack (4080 Super, 16GB VRAM)
Stages are sequential, not concurrent, so VRAM is freed between stages rather than shared across simultaneously-loaded models.

| Stage | Local option | Notes |
|---|---|---|
| Transcription | faster-whisper or whisper.cpp, model large-v3 | Fits comfortably, fast on 4080S |
| LLM (script/render plan) | Qwen2.5-14B-Instruct (GGUF, Q4/Q5) via Ollama | Comfortable fit, good instruction-following for JSON output |
| LLM (higher quality, optional) | Qwen2.5-32B (Q4 quant) via Ollama/llama.cpp | Fits tightly w/ offload, slower, better reasoning |
| TTS | Piper (fast/lightweight) or XTTS-v2/Coqui (higher quality, voice cloning) | XTTS needs a few GB VRAM but runs fine as its own stage |
| TTS fallback | edge-tts | Not local but free, no key, good quality |
| Render/export | ffmpeg with NVENC | GPU-accelerated encode, not model-bound |

`config/models.json` — makes the model backend swappable without touching pipeline code:
```json
{
  "llm": {
    "provider": "ollama",
    "model": "qwen2.5:14b-instruct",
    "endpoint": "http://localhost:11434/api/chat",
    "fallback_provider": "anthropic",
    "fallback_model": "claude-sonnet-4-6"
  },
  "transcription": {
    "provider": "faster-whisper",
    "model": "large-v3",
    "device": "cuda"
  },
  "tts": {
    "provider": "piper",
    "voice": "en_US-libritts-high",
    "fallback_provider": "edge-tts"
  }
}
```

Because local Qwen is less reliable at strict JSON than a frontier model, wrap the LLM call in a validate-and-retry loop: parse against the schema, and on failure re-prompt with the validation error appended (1–2 retries) before falling back if configured.

Load/unload discipline: `transcribe.py`, `plan_video.py`, `generate_voice.py` should each release their model before the next stage starts, so stages don't contend for the same 16GB.

## 6. Implementation Stages (build one at a time, confirm working before moving on)
Each stage is a self-contained checkpoint: build it, run it on a real test video, confirm the output looks/works right, then move to the next.

- **Stage 0 — Environment setup**
  - Install ffmpeg, Ollama (+ pull qwen2.5:14b-instruct), faster-whisper, Piper/XTTS, moviepy/OpenCV.
  - Set up folder structure + empty `config/models.json`.
  - Checkpoint: run each tool once manually (transcribe a test clip, ping Ollama, generate one TTS line).

- **Stage 1 — Ingest + transcription**
  - `ingest.py`: accept source video + metadata, write record to `data/`.
  - `transcribe.py`: extract audio, transcribe with faster-whisper, save word-level timestamped transcript.
  - Checkpoint: transcript JSON has accurate word-level timing on a real clip.

- **Stage 2 — Basic LLM render plan (no editing instructions yet)**
  - `plan_video.py`: LLM call producing just title, hook, script, clips.
  - Add validate-and-retry wrapper against schema.
  - Checkpoint: render plan JSON is valid, script reads naturally, clip timestamps make sense against the transcript.

- **Stage 3 — Voiceover**
  - `generate_voice.py`: TTS from script, then run a word-level forced-alignment pass over the generated audio and save word-level timestamps alongside it.
  - Checkpoint: voiceover sounds acceptable and word-level timestamps line up with what's actually spoken.

- **Stage 4 — Base render (no effects yet)**
  - `render.py`: source clips (hard cuts) sequenced with randomly-selected b-roll filling gaps, + voiceover + plain `.srt` subtitles, export vertical 1080x1920.
  - Checkpoint: a complete, watchable video comes out the other end — first true end-to-end run.

- **Stage 5 — Word-level karaoke captions + title card**
  - Build `.ass` generation from the Stage 3 word-level voiceover timestamps + `emphasis_words`.
  - Add caption presets (`bold_yellow_pop`, `minimal_white_bottom`, etc.) and title card renderer.
  - Checkpoint: captions highlight per word in sync with the voiceover; title card displays correctly.

- **Stage 6 — Editing instructions (zoom, transitions, effects, music cues)**
  - Extend `render.py` to read `zoom`, `transition_in`, `effects`, `music_cues`.
  - Build preset libraries (`effect_presets.json`), transitions via ffmpeg `xfade`.
  - Extend the LLM prompt to populate these fields.
  - Checkpoint: toggle each effect type on a test video and visually confirm it applies correctly.

- **Stage 7 — Review dashboard**
  - Streamlit/FastAPI: list drafts, show render plan + video side by side, approve/reject buttons.
  - Checkpoint: you can review and approve/reject a batch of drafts without touching the filesystem directly.

- **Stage 8 — Manual posting**
  - Approved videos land in `outputs/approved/` — you post them yourself.
  - Checkpoint: none needed — this is just you.
