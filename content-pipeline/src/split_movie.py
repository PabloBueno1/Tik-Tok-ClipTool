import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from pydantic import BaseModel, ValidationError

from chat_signal import flag_chat_spike_segments, load_chat_messages
from common import PENDING_CLIPS_DIR, call_llm, flag_loud_segments, load_config, probe_duration
from transcribe import extract_audio, transcribe_segments

MAX_ATTEMPTS = 3

DEFAULT_WINDOW_SECONDS = 10 * 60
DEFAULT_MIN_CLIP_SECONDS = 60
DEFAULT_MAX_CLIP_SECONDS = 90
DEFAULT_MAX_CLIPS_PER_WINDOW = 3
DEFAULT_MAX_TOTAL_CLIPS = 12
DEFAULT_MIN_SCORE = 5.0

# Final picks must sit at least this far apart - a weak scouting model loves
# nominating two adjacent cuts of the same fight, and two near-identical
# clips is always worse than the better one plus something from elsewhere.
MIN_CLIP_SPACING_SECONDS = 120.0

# Chat-anchored candidates: the top-N chat eruptions are guaranteed to
# produce a candidate even if the scouting LLM skipped that stretch - chat
# is measured audience ground truth, and it catches visual moments the
# transcript can't describe (where there may be little or no speech at all).
CHAT_ANCHOR_TOP_N = 8
CHAT_ANCHOR_SCORE = 7.0
# Seconds of clip kept after the eruption-causing moment (reaction tail);
# the rest of the clip length sits before it as buildup.
CHAT_ANCHOR_TAIL_SECONDS = 12.0
# How close (seconds) a real sentence boundary must be for an anchored clip's
# edge to snap to it instead of using the raw time.
EDGE_SNAP_TOLERANCE = 4.0

# Moment-centric re-cut: every final clip is re-centered on its strongest
# measured instant, giving buildup -> payoff-at-one-third -> aftermath, and a
# guaranteed cold-open teaser around that instant recorded in the manifest.
MOMENT_BUILDUP_SECONDS = 20.0
TEASER_PRE_SECONDS = 3.0
TEASER_POST_SECONDS = 1.5
# A teaser starting earlier than this into the clip would replay as the clip
# opens right after it (same rule the render's cold open enforces).
MIN_TEASER_OFFSET_SECONDS = 15.0

# Slack allowed ABOVE max_seconds after a clip's segment indices are
# auto-extended/trimmed, since real sentence boundaries rarely land exactly
# on the target lengths. min_seconds is a hard floor - platforms treat
# sub-minimum clips differently (and they read as low-effort), so never
# emit one under it.
LENGTH_TOLERANCE = 10.0

# Cap on the transcript excerpt stored per candidate (fed to the rerank pass).
EXCERPT_CHARS = 300

WINDOW_PROMPT_TEMPLATE = """You are scouting highlight moments in one window of a timestamped transcript from a longer stream VOD ({video_duration:.0f} seconds total). Each moment you pick will be cut into a standalone short-form clip.
{context_line}Each transcript segment below is numbered, with its time range. Two kinds of tags may appear:
- [LOUD REACTION]: the streamer's voice spiked well above normal there (yelling, laughing, hype).
- [CHAT ERUPTED]: the live chat replay suddenly flooded with messages and reaction emotes around that moment - the strongest signal available that many viewers found it clip-worthy, and it also catches purely visual moments the transcript can't describe.
Favor tagged moments when the content backs them up - [CHAT ERUPTED] especially. But tags are a bonus signal, not a requirement; plenty of great clips (deadpan jokes, good stories, clutch plays described calmly) carry no tag at all.

Pick up to {max_clips} moments from THIS WINDOW that could work as standalone clips: funny reactions or bits, exciting plays, surprising or insightful moments, drama, big wins/losses, good stories or banter. A later pass compares moments across the whole video and keeps only the best, so include any moment with real potential - but score it honestly.

Score each moment 1-10 for how well it would hold a stranger's attention on its own:
- 8-10: exceptional - big reaction, great punchline, jaw-dropping play, genuinely gripping story
- 5-7: solid - entertaining or interesting, would hold attention
- 1-4: weak - only include if the window has nothing better
Menu navigation, loadout/settings talk, queue or loading-screen chatter, routine gameplay narration with no payoff, and stream housekeeping (thanking subs, fixing audio, idly reading chat) are NOT highlights - skip them or score them 3 or below.

Respond with a JSON object: {{"clips": [...]}} (empty list if the whole window is filler). Each clip object has:
- start_index: integer - the number of the segment where the clip starts
- end_index: integer - the number of the segment where the clip ends (same as start_index or later)
- score: number 1-10 per the scale above
- reason: short string - why this moment works as a clip

Choose start_index/end_index so the spoken content spans roughly {min_seconds:.0f}-{max_seconds:.0f} seconds (use the time ranges shown on each segment). Timestamps are handled for you - only ever reference segments by their number.
Respond with ONLY the JSON object, no other text."""

RERANK_PROMPT_TEMPLATE = """You are picking the final lineup of short-form clips cut from one long stream VOD ({video_duration:.0f} seconds).
{context_line}Below are candidate moments already scouted from the transcript, each with an id, its time range, a first-pass score, the reason it was picked, and an excerpt of what is said. A "signals=" note means the moment had a MEASURED audience/streamer reaction: "chat" = the live chat erupted there (strong evidence of a clip-worthy moment), "loud" = the streamer's voice spiked.
IMPORTANT: the first-pass scores and reasons come from a weaker scouting model that inflates scores and often oversells mundane moments (and sometimes falsely claims chat reacted). Judge primarily from the quoted excerpt and the measured signals= flags, not from the claimed reason or score. An excerpt that reads as routine gameplay narration is routine gameplay narration, whatever the reason says.

Keep the best {max_total} - or fewer, if the rest are weak. Judge each on whether it would hold a stranger's attention as a standalone clip. Drop anything that reads like menu/setup talk, loading-screen filler, routine narration, or stream housekeeping. Prefer a varied lineup over several near-identical moments.

Respond with ONLY a JSON object: {{"keep": [id, id, ...]}} - the ids of the clips to keep."""


class HighlightClip(BaseModel):
    start_index: int
    end_index: int
    score: float
    reason: str


class HighlightWindow(BaseModel):
    clips: list[HighlightClip] = []


class RerankKeep(BaseModel):
    keep: list[int]


def format_context_line(context: str | None) -> str:
    return f"About this video: {context}\n" if context else ""


def format_segments_for_prompt(segments: list[dict]) -> str:
    lines = []
    for i, s in enumerate(segments):
        marker = " [LOUD REACTION]" if s.get("loud") else ""
        marker += " [CHAT ERUPTED]" if s.get("chat_spike") else ""
        lines.append(f"[{i}] [{s['start']:.1f}-{s['end']:.1f}]{marker} {s['text']}")
    return "\n".join(lines)


def bucket_segments(segments: list[dict], window_seconds: float) -> list[list[dict]]:
    """Groups transcript segments into fixed-size time windows by start time,
    so each LLM call only has to reason over a bounded amount of transcript -
    important for multi-hour stream VODs that would otherwise blow the LLM's
    context window."""
    buckets: dict[int, list[dict]] = {}
    for seg in segments:
        idx = int(seg["start"] // window_seconds)
        buckets.setdefault(idx, []).append(seg)
    return [buckets[k] for k in sorted(buckets)]


def resolve_clip(clip: HighlightClip, segments: list[dict], min_seconds: float, max_seconds: float) -> dict | None:
    """Turns an LLM-picked segment index range into concrete timestamps,
    auto-extending or trimming to neighboring segment boundaries so the clip
    lands inside the target length range. Cuts always sit on real sentence
    boundaries because timestamps only ever come from the transcript itself.
    Returns None if no boundary combination can fit the length range."""
    n = len(segments)
    if not (0 <= clip.start_index < n and 0 <= clip.end_index < n and clip.start_index <= clip.end_index):
        return None
    s_idx, e_idx = clip.start_index, clip.end_index

    def length() -> float:
        return segments[e_idx]["end"] - segments[s_idx]["start"]

    while length() < min_seconds and (e_idx < n - 1 or s_idx > 0):
        if e_idx < n - 1:
            e_idx += 1
        else:
            s_idx -= 1
    while length() > max_seconds and e_idx > s_idx:
        e_idx -= 1
    if not (min_seconds <= length() <= max_seconds + LENGTH_TOLERANCE):
        return None

    span = segments[s_idx : e_idx + 1]
    excerpt = " ".join(s["text"] for s in span)
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[: EXCERPT_CHARS - 3] + "..."
    return {
        "start": segments[s_idx]["start"],
        "end": segments[e_idx]["end"],
        "score": min(max(clip.score, 1.0), 10.0),
        "reason": clip.reason,
        "excerpt": excerpt,
        "loud": any(s.get("loud") for s in span),
        "chat_spike": any(s.get("chat_spike") for s in span),
    }


def generate_window_clips(
    window_segments: list[dict], video_duration: float, config: dict,
    min_seconds: float, max_seconds: float, max_clips: int, context: str | None,
) -> list[dict]:
    transcript_text = format_segments_for_prompt(window_segments)
    system_prompt = WINDOW_PROMPT_TEMPLATE.format(
        video_duration=video_duration, max_clips=max_clips,
        min_seconds=min_seconds, max_seconds=max_seconds,
        context_line=format_context_line(context),
    )
    messages = [
        {"role": "system", "content": "/no_think\n" + system_prompt},
        {"role": "user", "content": f"Transcript window:\n{transcript_text}"},
    ]

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = call_llm(messages, config)
        except requests.exceptions.RequestException as e:
            # Network/timeout hiccup talking to the local LLM - retry the same
            # request rather than burning an attempt on corrective feedback.
            last_error = e
            continue
        try:
            parsed = HighlightWindow.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That response was invalid: {e}\nPlease fix it and respond with ONLY the corrected JSON object.",
            })
            continue

        # Salvage per clip: one bad pick must never cost the window its good ones.
        clips = []
        for c in parsed.clips[:max_clips]:
            resolved = resolve_clip(c, window_segments, min_seconds, max_seconds)
            if resolved is None:
                print(
                    f"  dropping pick (segments {c.start_index}-{c.end_index}): "
                    f"invalid indices or can't fit {min_seconds:.0f}-{max_seconds:.0f}s on sentence boundaries",
                    file=sys.stderr,
                )
                continue
            clips.append(resolved)
        return clips

    print(
        f"warning: window [{window_segments[0]['start']:.1f}-{window_segments[-1]['end']:.1f}] "
        f"failed after {MAX_ATTEMPTS} attempts, skipping: {last_error}",
        file=sys.stderr,
    )
    return []


def candidate_rank_key(c: dict) -> tuple:
    """Measured signals outrank the scouting model's own score: a local LLM
    inflates scores and embellishes reasons, but a chat eruption or a vocal
    spike actually happened."""
    return (bool(c.get("chat_spike")), c["score"], bool(c.get("loud")))


def drop_overlaps(clips: list[dict], min_spacing: float = MIN_CLIP_SPACING_SECONDS) -> list[dict]:
    """Greedily keeps the best clips (chat-backed first, then score), dropping
    any that overlap or sit within min_spacing of an already-kept one, then
    returns the survivors in timeline order."""
    kept: list[dict] = []
    for c in sorted(clips, key=candidate_rank_key, reverse=True):
        if all(c["end"] + min_spacing <= k["start"] or c["start"] >= k["end"] + min_spacing for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c["start"])


def anchor_chat_candidates(
    spikes: list[dict], segments: list[dict], candidates: list[dict],
    min_seconds: float, max_seconds: float, top_n: int = CHAT_ANCHOR_TOP_N,
) -> list[dict]:
    """Builds a candidate around each of the biggest chat eruptions that no
    scouted candidate already covers. The eruption moment lands near the end
    of the clip (buildup -> payoff -> reaction tail). Edges snap to nearby
    sentence boundaries when one is close; otherwise raw times are used -
    eruptions over silent gameplay are exactly the visual moments the
    transcript-based scout can't see."""
    anchored: list[dict] = []
    video_end = max(s["end"] for s in segments)
    target = (min_seconds + max_seconds) / 2
    for spike in sorted(spikes, key=lambda s: s["weight"], reverse=True)[:top_n]:
        t = spike["time"]
        if any(c["start"] <= t <= c["end"] for c in candidates + anchored):
            continue
        end = min(t + CHAT_ANCHOR_TAIL_SECONDS, video_end)
        start = max(end - target, 0.0)
        # Snap edges to a nearby real sentence boundary when one is close.
        starts = [s["start"] for s in segments]
        ends = [s["end"] for s in segments]
        near_start = min(starts, key=lambda b: abs(b - start))
        near_end = min(ends, key=lambda b: abs(b - end))
        if abs(near_start - start) <= EDGE_SNAP_TOLERANCE and near_start < t:
            start = near_start
        if abs(near_end - end) <= EDGE_SNAP_TOLERANCE and near_end > t:
            end = near_end
        if not (min_seconds <= end - start <= max_seconds + LENGTH_TOLERANCE):
            continue
        span_texts = [s["text"] for s in segments if s["start"] < end and s["end"] > start]
        excerpt = (" ".join(span_texts) or "(little/no speech - likely a visual moment)")[:EXCERPT_CHARS]
        anchored.append({
            "start": start,
            "end": end,
            "score": CHAT_ANCHOR_SCORE,
            "reason": f"chat erupted here (weighted reaction volume {spike['weight']:.0f}) - auto-anchored, not LLM-picked",
            "excerpt": excerpt,
            "loud": any(s.get("loud") for s in segments if s["start"] < end and s["end"] > start),
            "chat_spike": True,
        })
    return anchored


def rerank_candidates(
    candidates: list[dict], config: dict, video_duration: float, max_total: int, context: str | None,
) -> list[dict]:
    """Second LLM pass over every surviving candidate from every window: with
    the whole video's picks side by side it keeps the strongest max_total and
    drops filler that looked passable in its own window. Falls back to a plain
    score sort if the LLM can't produce a usable answer."""
    if len(candidates) <= max_total:
        return candidates

    lines = []
    for i, c in enumerate(candidates):
        signals = "+".join(name for name, on in (("chat", c.get("chat_spike")), ("loud", c.get("loud"))) if on)
        signals_note = f" signals={signals}" if signals else ""
        lines.append(
            f"[{i}] {c['start']:.0f}-{c['end']:.0f}s ({c['end'] - c['start']:.0f}s) "
            f"score={c['score']:.0f}{signals_note} | {c['reason']} | \"{c['excerpt']}\""
        )
    system_prompt = RERANK_PROMPT_TEMPLATE.format(
        video_duration=video_duration, max_total=max_total,
        context_line=format_context_line(context),
    )
    messages = [
        {"role": "system", "content": "/no_think\n" + system_prompt},
        {"role": "user", "content": "Candidates:\n" + "\n".join(lines)},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = call_llm(messages, config)
        except requests.exceptions.RequestException:
            continue
        try:
            parsed = RerankKeep.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That response was invalid: {e}\nPlease fix it and respond with ONLY the corrected JSON object.",
            })
            continue
        keep = [i for i in dict.fromkeys(parsed.keep) if 0 <= i < len(candidates)]
        if keep:
            return sorted((candidates[i] for i in keep[:max_total]), key=lambda c: c["start"])

    print("warning: rerank pass failed, falling back to keeping the top-ranked candidates", file=sys.stderr)
    return sorted(
        sorted(candidates, key=candidate_rank_key, reverse=True)[:max_total],
        key=lambda c: c["start"],
    )


def find_clip_moment(clip: dict, spikes: list[dict], segments: list[dict]) -> float:
    """The clip's strongest measured instant (in VOD time): the biggest chat
    eruption inside it, else the center of its longest loud-reaction run,
    else just past the clip's midpoint (so a teaser still exists)."""
    inside = [s for s in spikes if clip["start"] <= s["time"] <= clip["end"]]
    if inside:
        return max(inside, key=lambda s: s["weight"])["time"]

    runs: list[list[dict]] = []
    run: list[dict] = []
    for seg in segments:
        if seg["end"] < clip["start"] or seg["start"] > clip["end"]:
            continue
        if seg.get("loud"):
            run.append(seg)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    if runs:
        best = max(runs, key=lambda r: r[-1]["end"] - r[0]["start"])
        return (best[0]["start"] + best[-1]["end"]) / 2

    return (clip["start"] + clip["end"]) / 2


def recenter_on_moment(
    clip: dict, moment: float, video_duration: float,
    min_seconds: float, max_seconds: float, segments: list[dict],
) -> dict:
    """Rebuilds the clip window around its peak moment - buildup before,
    payoff at roughly the one-third mark, aftermath after - and attaches the
    clip-local teaser range. The teaser is dropped (None) only when the
    moment sits so close to the video's start that the buildup can't exist."""
    length = min(max(clip["end"] - clip["start"], min_seconds), max_seconds)
    start = moment - MOMENT_BUILDUP_SECONDS
    start = max(min(start, video_duration - length), 0.0)
    end = min(start + length, video_duration)

    # Nudge edges onto nearby sentence boundaries when that keeps the length
    # legal and the payoff far enough in.
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    near_start = min(starts, key=lambda b: abs(b - start))
    if abs(near_start - start) <= EDGE_SNAP_TOLERANCE and moment - near_start >= MIN_TEASER_OFFSET_SECONDS + TEASER_PRE_SECONDS:
        candidate_end = min(near_start + length, video_duration)
        if candidate_end - near_start >= min_seconds:
            start, end = near_start, candidate_end
    near_end = min(ends, key=lambda b: abs(b - end))
    if abs(near_end - end) <= EDGE_SNAP_TOLERANCE and min_seconds <= near_end - start <= max_seconds + LENGTH_TOLERANCE and near_end > moment + TEASER_POST_SECONDS:
        end = near_end

    teaser = None
    t_start = moment - TEASER_PRE_SECONDS - start
    t_end = min(moment + TEASER_POST_SECONDS, end) - start
    if t_start >= MIN_TEASER_OFFSET_SECONDS and t_end - t_start >= 1.0:
        teaser = {"start": round(t_start, 3), "end": round(t_end, 3)}

    span = [s for s in segments if s["start"] < end and s["end"] > start]
    excerpt = " ".join(s["text"] for s in span)
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[: EXCERPT_CHARS - 3] + "..."

    return {
        **clip,
        "start": start,
        "end": end,
        "moment": moment,
        "teaser": teaser,
        "excerpt": excerpt or clip.get("excerpt", ""),
        "loud": any(s.get("loud") for s in span),
        "chat_spike": any(s.get("chat_spike") for s in span),
    }


def cut_clip(video_path: Path, start: float, end: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip cut failed:\n{result.stderr}")


def split_into_highlights(
    video_path: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
    max_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    max_clips_per_window: int = DEFAULT_MAX_CLIPS_PER_WINDOW,
    max_total_clips: int = DEFAULT_MAX_TOTAL_CLIPS,
    min_score: float = DEFAULT_MIN_SCORE,
    context: str | None = None,
    chat_path: Path | None = None,
) -> list[Path]:
    """Transcribes a long-form video, has the LLM scout scored highlight
    moments per transcript window (picking segment index ranges, so cuts
    always land on real sentence boundaries), then reranks all candidates
    globally and keeps the best max_total_clips. Chunks land in
    PENDING_CLIPS_DIR, the staging folder run_pipeline.py's batch mode loops
    over."""
    duration = probe_duration(video_path)
    if duration <= max_seconds:
        return [video_path]

    # Whisper on a multi-hour VOD is the slowest stage by far - cache the
    # (loudness-flagged) transcript next to the video so selection logic can
    # be iterated on without re-transcribing. Chat flags are recomputed each
    # run on purpose; that analysis is cheap and still evolving.
    cache_path = video_path.with_name(video_path.stem + ".transcript.json")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        print(f"loaded cached transcript ({len(segments)} segments) from {cache_path.name}")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            extract_audio(video_path, audio_path)
            segments, _info = transcribe_segments(audio_path)
            flag_loud_segments(audio_path, segments)
        for seg in segments:
            seg.pop("words", None)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(segments, f)
        print(f"cached transcript to {cache_path.name}")

    if not segments:
        raise RuntimeError(f"No speech detected in {video_path} - can't semantically split.")

    if chat_path is not None and not chat_path.exists():
        raise FileNotFoundError(f"Chat replay file not found: {chat_path}")
    chat_file = chat_path or video_path.with_name(video_path.stem + ".live_chat.json")
    spikes: list[dict] = []
    if chat_file.exists():
        chat_messages = load_chat_messages(chat_file)
        flagged, spikes = flag_chat_spike_segments(chat_messages, segments)
        print(f"chat replay: {len(chat_messages)} messages from {chat_file.name}, {flagged} segment(s) tagged [CHAT ERUPTED], {len(spikes)} eruption(s)")
    else:
        print(
            "no chat replay found (looked for --chat or a .live_chat.json next to the video) - "
            "selection runs on transcript + audio only",
        )

    llm_config = load_config()["llm"]
    windows = bucket_segments(segments, window_seconds)

    candidates = []
    for i, window_segments in enumerate(windows, start=1):
        clips = generate_window_clips(
            window_segments, duration, llm_config, min_seconds, max_seconds, max_clips_per_window, context,
        )
        print(
            f"window {i}/{len(windows)} [{window_segments[0]['start']:.0f}-{window_segments[-1]['end']:.0f}s]: "
            f"{len(clips)} candidate(s)"
        )
        candidates.extend(clips)

    if spikes:
        anchored = anchor_chat_candidates(spikes, segments, candidates, min_seconds, max_seconds)
        if anchored:
            print(f"anchored {len(anchored)} extra candidate(s) directly on chat eruptions the scout missed")
        candidates.extend(anchored)

    strong = [c for c in candidates if c["score"] >= min_score]
    if len(strong) < len(candidates):
        print(f"dropped {len(candidates) - len(strong)} candidate(s) scoring below {min_score:.0f}")
    strong = drop_overlaps(strong)

    accepted = rerank_candidates(strong, llm_config, duration, max_total_clips, context)
    accepted = [
        recenter_on_moment(c, find_clip_moment(c, spikes, segments), duration, min_seconds, max_seconds, segments)
        for c in accepted
    ]
    accepted = drop_overlaps(accepted, min_spacing=0.0)
    with_teaser = sum(1 for c in accepted if c["teaser"])
    print(
        f"{len(candidates)} candidates -> {len(accepted)} final clips after scoring, overlap and rerank passes "
        f"({with_teaser} with a measured cold-open teaser)"
    )
    if not accepted:
        raise RuntimeError(
            f"No highlight candidates in {video_path} scored {min_score:.0f}+ - "
            f"lower --min-score or check the transcript quality."
        )

    PENDING_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    output_paths = []
    for i, clip in enumerate(accepted, start=1):
        out_path = PENDING_CLIPS_DIR / f"{video_path.stem}_{i:02d}.mp4"
        cut_clip(video_path, clip["start"], clip["end"], out_path)
        output_paths.append(out_path)
        manifest.append({
            "file": out_path.name,
            "source_start": clip["start"],
            "source_end": clip["end"],
            "moment": clip["moment"],
            "teaser": clip["teaser"],
            "score": clip["score"],
            "reason": clip["reason"],
            "excerpt": clip["excerpt"],
            "loud": clip["loud"],
            "chat_spike": clip["chat_spike"],
        })

    manifest_path = PENDING_CLIPS_DIR / f"{video_path.stem}_highlights.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_video": str(video_path.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clips": manifest,
        }, f, indent=2)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Semantically split a long-form video into highlight clips using transcript + LLM moment selection.")
    parser.add_argument("--video", required=True, help="Path to the long-form source video.")
    parser.add_argument("--context", default=None, help="One-line description of the video (streamer, game, event) given to the LLM to judge moments better.")
    parser.add_argument("--chat", default=None, help="Path to a chat replay file (yt-dlp .live_chat.json or chat-downloader JSON). Default: auto-detect <video>.live_chat.json next to the video.")
    parser.add_argument("--window-minutes", type=float, default=DEFAULT_WINDOW_SECONDS / 60, help="Transcript window size fed to the LLM per call (default: 15).")
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_CLIP_SECONDS, help="Minimum clip length in seconds (default: 60).")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS, help="Maximum clip length in seconds (default: 90).")
    parser.add_argument("--max-clips-per-window", type=int, default=DEFAULT_MAX_CLIPS_PER_WINDOW, help="Max highlight candidates the LLM may pick per window (default: 4).")
    parser.add_argument("--max-total-clips", type=int, default=DEFAULT_MAX_TOTAL_CLIPS, help="Max clips kept for the whole video after the global rerank pass (default: 12).")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Candidates scoring below this (1-10) are dropped before reranking (default: 5).")
    args = parser.parse_args()

    for chunk_path in split_into_highlights(
        Path(args.video),
        window_seconds=args.window_minutes * 60,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        max_clips_per_window=args.max_clips_per_window,
        max_total_clips=args.max_total_clips,
        min_score=args.min_score,
        context=args.context,
        chat_path=Path(args.chat) if args.chat else None,
    ):
        print(chunk_path)


if __name__ == "__main__":
    main()
