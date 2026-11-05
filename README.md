# TikiToki

A local pipeline that turns long stream VODs into vertical short-form clips (TikTok / Shorts / Reels). Everything runs on your own machine: transcription, clip selection, hooks, voiceover, captions and rendering.

## How it works

1. **Split.** A long video is transcribed with faster-whisper. A local LLM (via Ollama) scores transcript windows for highlight moments, using loudness and chat-replay spikes as extra signals, then reranks candidates across the whole VOD.
2. **Plan.** For each clip the LLM writes a short hook (best of 5) and picks a cold-open teaser built around the clip's peak moment.
3. **Render.** ffmpeg renders a 9:16 video in one of two modes:
   - **Raw mode** (`--raw`): full-frame clip with the streamer's own audio, word-by-word karaoke captions, hook overlay and cold-open teaser.
   - **Narrated mode** (default): TTS voiceover (Piper, with edge-tts as fallback) over a split screen with gameplay b-roll.
4. **Review.** Drafts land in `outputs/drafts/`. Move the ones you like to `outputs/approved/` and post them yourself.

See [content-pipeline-design.md](content-pipeline-design.md) for the full design.

## Requirements

- Python 3.12
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
- [Ollama](https://ollama.com/) running locally with the model from `config/models.json` (default `qwen3:14b`)
- An NVIDIA GPU is recommended. Transcription defaults to `large-v3` on CUDA; change `config/models.json` to run on CPU.
- To download YouTube VODs with yt-dlp you also need a JavaScript runtime such as [Deno](https://deno.com/) on your `PATH`.

## Setup

```bash
cd content-pipeline
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
python src/download_voices.py   # fetches the Piper voice models (~400 MB)
```

## Usage

```bash
# Download a VOD together with its chat replay (optional, but it improves clip selection)
yt-dlp -o "inputs/source_videos/%(id)s.%(ext)s" --sub-langs live_chat --write-subs <VOD_URL>

# Split it into highlights and render them in raw mode
python src/run_pipeline.py --video inputs/source_videos/<id>.mp4 --title "<streamer / game context>" --raw
```

Run `python src/run_pipeline.py --help` to see the splitting options (clip length, number of clips, minimum score, chat file).
