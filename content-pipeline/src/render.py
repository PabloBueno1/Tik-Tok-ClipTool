import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

from common import (
    CONFIG_DIR,
    DRAFTS_DIR,
    GAMEPLAY_LOOPS_DIR,
    RENDER_PLANS_DIR,
    VIDEO_EXTENSIONS,
    VOICE_DIR,
    load_effect_presets,
    load_ingest_record,
    probe_duration,
)

FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920
HALF_HEIGHT = FRAME_HEIGHT // 2

DEFAULT_CROSSFADE_DURATION = 0.4

MAX_CUE_WORDS = 8
MAX_CUE_DURATION = 3.5

CAPTION_PRESETS_PATH = CONFIG_DIR / "caption_presets.json"
DEFAULT_CAPTION_PRESET = "bold_yellow_pop"

HOOK_DURATION_DEFAULT = 3.5
HOOK_FADE_IN_MS = 250
HOOK_FADE_OUT_MS = 200
HOOK_POP_MS = 300
HOOK_START_SCALE = 70
HOOK_MARGIN_V = 120

CAM_BUBBLE_WIDTH = 340
CAM_MARGIN_PX = 24

VOICE_LEVEL = 0.25
SOURCE_AUDIO_DUCK = 0.75

def load_render_plan(record_id: str) -> dict:
    path = RENDER_PLANS_DIR / f"{record_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_voice_words(record_id: str) -> list[dict]:
    path = VOICE_DIR / f"{record_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["words"]


def load_caption_preset(name: str) -> dict:
    with open(CAPTION_PRESETS_PATH, "r", encoding="utf-8") as f:
        presets = json.load(f)
    if name not in presets:
        raise ValueError(f"Unknown caption preset '{name}'. Available: {list(presets)}")
    return presets[name]


def plan_source_clips(video_path: Path, clips: list[dict]) -> list[dict]:
    """Returns a list of clip dicts: start, duration, zoom, transition_in."""
    if not clips:
        raise RuntimeError("Render plan has no clips to build the source half from.")

    source_duration = probe_duration(video_path)
    valid = []
    for c in clips:
        start, end = c["source_start"], min(c["source_end"], source_duration)
        if end <= start:
            continue
        valid.append({
            "start": start,
            "duration": end - start,
            "zoom": c.get("zoom"),
            "transition_in": c.get("transition_in", "hard_cut") if valid else "hard_cut",
        })

    if not valid:
        raise RuntimeError("All clips in the render plan fell outside the source video's duration.")

    return valid


def crossfade_covered_duration(clips: list[dict], crossfade_duration: float) -> float:
    """Total on-screen duration of the concatenated top half, accounting for
    crossfade transitions shortening the combined length by the overlap."""
    covered = clips[0]["duration"]
    for clip in clips[1:]:
        overlap = crossfade_duration if clip["transition_in"] != "hard_cut" else 0.0
        covered += clip["duration"] - overlap
    return covered


def fit_clips_to_duration(clips: list[dict], target_duration: float, crossfade_duration: float, source_duration: float) -> list[dict]:
    """Grows or shrinks clips (from the last one backward) so the concatenated/
    crossfaded top-half footage covers exactly target_duration - real
    contiguous source footage extended or trimmed, never frozen or looped."""
    clips = [dict(c) for c in clips]
    diff = target_duration - crossfade_covered_duration(clips, crossfade_duration)

    if abs(diff) < 0.01:
        return clips

    if diff > 0:
        remaining = diff
        for clip in reversed(clips):
            if remaining <= 0:
                break
            room = max(source_duration - (clip["start"] + clip["duration"]), 0)
            grow = min(remaining, room)
            clip["duration"] += grow
            remaining -= grow
        if remaining > 0.01:
            # No room left after any clip - fall back to pulling the first
            # clip's start earlier, grabbing unused footage before it.
            first = clips[0]
            room = first["start"]
            grow = min(remaining, room)
            if grow > 0:
                first["start"] -= grow
                first["duration"] += grow
                remaining -= grow
        if remaining > 0.01:
            raise RuntimeError(
                f"Source video doesn't have enough footage around the picked clips to cover "
                f"the {target_duration:.1f}s voiceover (short by {remaining:.1f}s)."
            )
    else:
        remaining = -diff
        for clip in reversed(clips):
            if remaining <= 0:
                break
            shrink = min(remaining, max(clip["duration"] - 0.1, 0))
            clip["duration"] -= shrink
            remaining -= shrink
        if remaining > 0.01:
            raise RuntimeError(
                f"Picked clips are still {remaining:.1f}s longer than the {target_duration:.1f}s "
                f"voiceover even at minimum length."
            )

    return clips


def plan_broll_chunks(total_duration: float) -> list[tuple[Path, float, float]]:
    """Bottom half: gameplay b-roll, playing continuously for the whole video
    regardless of what's happening in the source half. One random loop file
    and one random start offset are picked per render, then it plays unbroken
    for the full duration - continuous gameplay reads far more natural than
    rapid jump-cuts to random points within the same footage. Returns
    [(path, start, duration), ...]."""
    loop_files = [p for p in sorted(GAMEPLAY_LOOPS_DIR.glob("*")) if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not loop_files:
        raise RuntimeError(f"No gameplay loop files found in {GAMEPLAY_LOOPS_DIR}")

    path = random.choice(loop_files)
    clip_duration = probe_duration(path)

    if clip_duration >= total_duration:
        max_start = clip_duration - total_duration
        start = random.uniform(0, max_start) if max_start > 0 else 0
        return [(path, start, total_duration)]

    # Loop file is shorter than the render (rare) - play it start to end,
    # then wrap back to the start to cover the remainder.
    chunks = [(path, 0.0, clip_duration)]
    covered = clip_duration
    while covered < total_duration:
        chunk_len = min(clip_duration, total_duration - covered)
        chunks.append((path, 0.0, chunk_len))
        covered += chunk_len

    return chunks


def fit_filter(label_in: str, label_out: str) -> str:
    return (
        f"[{label_in}]scale={FRAME_WIDTH}:{HALF_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={FRAME_WIDTH}:{HALF_HEIGHT},setsar=1,setpts=PTS-STARTPTS[{label_out}]"
    )


def build_clip_filter(input_index: int, clip: dict, label_out: str, cam_region: dict | None) -> str:
    """Per-clip filter chain: fit to the top half -> optional zoom (Ken
    Burns) -> optional cam bubble overlay (the streamer's webcam,
    auto-detected at ingest time, cropped from the same input and composited
    in a corner so it isn't lost to the gameplay-centered main crop)."""
    main_src = f"c{input_index}main" if cam_region else f"{input_index}:v"
    parts = []
    if cam_region:
        parts.append(f"[{input_index}:v]split=2[{main_src}][c{input_index}camsrc]")

    parts.append(
        f"[{main_src}]scale={FRAME_WIDTH}:{HALF_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={FRAME_WIDTH}:{HALF_HEIGHT},setsar=1,setpts=PTS-STARTPTS[c{input_index}f]"
    )
    last = f"c{input_index}f"

    if clip["zoom"]:
        d = max(clip["duration"], 0.01)
        z = f"{clip['zoom']['start_scale']}+({clip['zoom']['end_scale']}-{clip['zoom']['start_scale']})*min(t/{d:.3f},1)"
        parts.append(f"[{last}]scale=w='{FRAME_WIDTH}*({z})':h='{HALF_HEIGHT}*({z})':eval=frame[c{input_index}z]")
        parts.append(
            f"[c{input_index}z]crop={FRAME_WIDTH}:{HALF_HEIGHT}:"
            f"x='(in_w-{FRAME_WIDTH})/2':y='(in_h-{HALF_HEIGHT})/2'[c{input_index}zc]"
        )
        last = f"c{input_index}zc"

    if cam_region:
        cam_x = f"iw*{cam_region['x_ratio']:.4f}"
        cam_y = f"ih*{cam_region['y_ratio']:.4f}"
        cam_w = f"iw*{cam_region['w_ratio']:.4f}"
        cam_h = f"ih*{cam_region['h_ratio']:.4f}"
        parts.append(
            f"[c{input_index}camsrc]crop=w='{cam_w}':h='{cam_h}':x='{cam_x}':y='{cam_y}',"
            f"scale={CAM_BUBBLE_WIDTH}:-2[c{input_index}cam]"
        )
        parts.append(
            f"[{last}][c{input_index}cam]overlay=x=main_w-overlay_w-{CAM_MARGIN_PX}:y=main_h-overlay_h-{CAM_MARGIN_PX}[c{input_index}camout]"
        )
        last = f"c{input_index}camout"

    parts.append(f"[{last}]fps=30[{label_out}]")
    return ";".join(parts)


def build_source_audio_filter(clip_indices: list[int], source_clips: list[dict], crossfade_duration: float, label_out: str) -> list[str]:
    """Mirrors the video top-half concat/crossfade chain so the original clip
    audio (game sound, the streamer's real voice/reactions) stays in sync
    with it, ready to be ducked under the voiceover."""
    filters = [f"[{clip_indices[0]}:a]asetpts=PTS-STARTPTS[a0]"]
    running = "a0"
    for i in range(1, len(clip_indices)):
        clip = source_clips[i]
        next_label = f"a{i}raw"
        combo = f"asrcrun{i}"
        filters.append(f"[{clip_indices[i]}:a]asetpts=PTS-STARTPTS[{next_label}]")
        if clip["transition_in"] == "hard_cut":
            filters.append(f"[{running}][{next_label}]concat=n=2:v=0:a=1[{combo}]")
        else:
            filters.append(f"[{running}][{next_label}]acrossfade=d={crossfade_duration:.3f}[{combo}]")
        running = combo
    filters.append(f"[{running}]anull[{label_out}]")
    return filters


def format_ass_timestamp(seconds: float) -> str:
    total_cs = int(round(max(seconds, 0) * 100))
    h, total_cs = divmod(total_cs, 360000)
    m, total_cs = divmod(total_cs, 6000)
    s, cs = divmod(total_cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def ass_override_color(style_color: str) -> str:
    """Style-line colors are &HAABBGGRR; inline \\c override tags want &HBBGGRR&."""
    bbggrr = style_color.replace("&H", "")[-6:]
    return f"&H{bbggrr}&"


def group_caption_cues(words: list[dict]) -> list[list[dict]]:
    cues = []
    chunk = []
    for w in words:
        chunk.append(w)
        if len(chunk) >= MAX_CUE_WORDS or (chunk[-1]["end"] - chunk[0]["start"]) >= MAX_CUE_DURATION:
            cues.append(chunk)
            chunk = []
    if chunk:
        cues.append(chunk)
    return cues


def build_ass(words: list[dict], preset: dict, hook_text: str, hook_duration: float, caption_offset: float = 0.0) -> str:
    """Word-level karaoke captions: each cue's full text stays on screen while
    the currently-spoken word pops in scale/color, plus the hook text overlaid
    top-center for the first hook_duration seconds while the video plays
    underneath (no held intro card - content starts at frame zero).
    caption_offset shifts the caption cues (not the hook) later, matching a
    voiceover delayed behind a cold-open teaser."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {FRAME_WIDTH}
PlayResY: {FRAME_HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{preset['font_name']},{preset['font_size']},{preset['primary_color']},&H00000000,{preset['outline_color']},{preset['back_color']},{-1 if preset['bold'] else 0},0,0,0,100,100,0,0,1,{preset['outline']},{preset['shadow']},{preset['alignment']},40,40,{preset['margin_v']},1
Style: Hook,{preset['font_name']},{int(preset['font_size'] * 1.15)},&H00FFFFFF,&H00000000,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,0,8,60,60,{HOOK_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]

    hook_tag = (
        f"{{\\fad({HOOK_FADE_IN_MS},{HOOK_FADE_OUT_MS})"
        f"\\t(0,{HOOK_POP_MS},\\fscx100\\fscy100)\\fscx{HOOK_START_SCALE}\\fscy{HOOK_START_SCALE}}}"
    )
    lines.append(
        f"Dialogue: 0,{format_ass_timestamp(0)},{format_ass_timestamp(hook_duration)},Hook,,0,0,0,,"
        f"{hook_tag}{escape_ass_text(hook_text)}\n"
    )

    highlight = ass_override_color(preset["highlight_color"])
    scale_pop = int(preset["scale_pop"] * 100)

    for cue in group_caption_cues(words):
        for i, word in enumerate(cue):
            start = word["start"] + caption_offset
            end = (cue[i + 1]["start"] if i + 1 < len(cue) else word["end"]) + caption_offset
            if end <= start:
                continue
            parts = []
            for j, w in enumerate(cue):
                token = escape_ass_text(w["word"])
                if j == i:
                    parts.append(f"{{\\c{highlight}\\fscx{scale_pop}\\fscy{scale_pop}}}{token}{{\\r}}")
                else:
                    parts.append(token)
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{format_ass_timestamp(start)},{format_ass_timestamp(end)},Caption,,0,0,0,,{text}\n"
            )

    return "".join(lines)


def run_ffmpeg(
    voice_path: Path,
    source_video: Path,
    source_clips: list[dict],
    broll_chunks: list[tuple[Path, float, float]],
    presets: dict,
    ass_path: Path,
    total_duration: float,
    out_path: Path,
    cam_region: dict | None = None,
    voice_delay: float = 0.0,
) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(voice_path)]

    input_index = 1
    clip_indices = []
    for clip in source_clips:
        cmd += ["-ss", f"{clip['start']:.3f}", "-t", f"{clip['duration']:.3f}", "-i", str(source_video)]
        clip_indices.append(input_index)
        input_index += 1

    broll_indices = []
    for path, start, duration in broll_chunks:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path)]
        broll_indices.append(input_index)
        input_index += 1

    filters = []

    crossfade_duration = presets.get("crossfade_duration", DEFAULT_CROSSFADE_DURATION)
    for i, (idx, clip) in enumerate(zip(clip_indices, source_clips)):
        filters.append(build_clip_filter(idx, clip, f"clip{i}", cam_region))

    top_source_label = "clip0"
    running_duration = source_clips[0]["duration"]
    for i in range(1, len(source_clips)):
        clip = source_clips[i]
        next_label = f"clip{i}"
        combo_raw = f"srcrun{i}raw"
        combo_label = f"srcrun{i}"
        if clip["transition_in"] == "hard_cut":
            filters.append(f"[{top_source_label}][{next_label}]concat=n=2:v=1:a=0[{combo_raw}]")
            running_duration += clip["duration"]
        else:
            offset = running_duration - crossfade_duration
            filters.append(
                f"[{top_source_label}][{next_label}]xfade=transition={clip['transition_in']}:"
                f"duration={crossfade_duration:.3f}:offset={offset:.3f}[{combo_raw}]"
            )
            running_duration += clip["duration"] - crossfade_duration
        # xfade/concat don't reliably preserve the 1/30 timebase set by fps=30 above,
        # so re-normalize before this stream feeds the next transition in the chain.
        filters.append(f"[{combo_raw}]fps=30[{combo_label}]")
        top_source_label = combo_label

    filters.append(f"[{top_source_label}]null[top]")

    # concat requires every input to already share the same resolution/SAR, so each
    # b-roll chunk (loop files can be different native resolutions) is normalized
    # individually before concatenating rather than fitting the joined result after.
    broll_labels = []
    for j, idx in enumerate(broll_indices):
        filters.append(fit_filter(f"{idx}:v", f"broll{j}"))
        broll_labels.append(f"[broll{j}]")
    filters.append(f"{''.join(broll_labels)}concat=n={len(broll_indices)}:v=1:a=0[bottom]")

    filters.append("[top][bottom]vstack=inputs=2[mainraw]")

    ass_arg = str(ass_path).replace("\\", "/").replace(":", "\\:")
    filters.append(f"[mainraw]ass='{ass_arg}',format=yuv420p,fps=30[vout]")

    filters += build_source_audio_filter(clip_indices, source_clips, crossfade_duration, "srcaud")
    filters.append(f"[srcaud]volume={SOURCE_AUDIO_DUCK}[srcauddim]")
    voice_chain = f"adelay={int(round(voice_delay * 1000))}:all=1," if voice_delay > 0 else ""
    filters.append(f"[0:a]{voice_chain}volume={VOICE_LEVEL}[voicedim]")
    filters.append("[voicedim][srcauddim]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc",
        "-c:a", "aac",
        "-t", f"{voice_delay + total_duration:.3f}",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed:\n{result.stderr}")


def render(record_id: str) -> Path:
    plan = load_render_plan(record_id)
    ingest_record = load_ingest_record(record_id)
    words = load_voice_words(record_id)

    voice_path = VOICE_DIR / f"{record_id}.wav"
    total_duration = probe_duration(voice_path)

    presets = load_effect_presets()
    source_video = Path(ingest_record["video_path"])
    source_clips = plan_source_clips(source_video, plan["clips"])
    crossfade_duration = presets.get("crossfade_duration", DEFAULT_CROSSFADE_DURATION)
    source_clips = fit_clips_to_duration(source_clips, total_duration, crossfade_duration, probe_duration(source_video))

    # Cold-open teaser: the plan's peak moment plays first (raw audio, hook
    # text on top), then the main sequence hard-cuts in from the beginning
    # while the voiceover starts. A missing/repaired-away hook_clip simply
    # means no cold open.
    cold_open = plan.get("hook_clip")
    cold_open_duration = 0.0
    if cold_open:
        cold_open_duration = cold_open["source_end"] - cold_open["source_start"]
        source_clips = [{
            "start": cold_open["source_start"],
            "duration": cold_open_duration,
            "zoom": None,
            "transition_in": "hard_cut",
        }] + source_clips

    broll_chunks = plan_broll_chunks(cold_open_duration + total_duration)

    caption_style = plan.get("caption_style", {})
    preset = load_caption_preset(caption_style.get("preset", DEFAULT_CAPTION_PRESET))

    hook_cfg = plan.get("title_card", {})
    hook_text = hook_cfg.get("text") or plan["hook"]
    hook_duration = min(
        cold_open_duration + hook_cfg.get("duration", HOOK_DURATION_DEFAULT),
        cold_open_duration + total_duration,
    )

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    ass_path = DRAFTS_DIR / f"{record_id}.ass"
    ass_path.write_text(
        build_ass(words, preset, hook_text, hook_duration, caption_offset=cold_open_duration),
        encoding="utf-8",
    )

    final_path = DRAFTS_DIR / f"{record_id}.mp4"
    run_ffmpeg(
        voice_path=voice_path,
        source_video=source_video,
        source_clips=source_clips,
        broll_chunks=broll_chunks,
        presets=presets,
        ass_path=ass_path,
        total_duration=total_duration,
        out_path=final_path,
        cam_region=ingest_record.get("cam_region"),
        voice_delay=cold_open_duration,
    )

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Render the split-screen video with voiceover, karaoke captions, and a hook text overlay.")
    parser.add_argument("--id", required=True, help="Pipeline record id (shared across ingest/transcribe/plan/voice).")
    args = parser.parse_args()

    print(render(args.id))


if __name__ == "__main__":
    main()
