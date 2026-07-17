import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np
import requests
from pydantic import BaseModel, ValidationError

from common import RENDER_PLANS_DIR, SCRIPTS_DIR, TRANSCRIPTS_DIR, call_llm, flag_loud_segments, load_config, load_ingest_record

MAX_ATTEMPTS = 5
# The rendered video's length IS the voiceover's length (plus a few seconds
# of cold open), so the script floor is what guarantees a 60s+ final video.
# 170 words at ~155 wpm reads out around 65s - comfortably over the minute
# mark that platforms (and viewers) treat as a real clip.
MIN_SCRIPT_WORDS = 170
WORDS_PER_MINUTE = 155
# Buffer (seconds) beyond the clip's own duration that fit_clips_to_duration
# can realistically stretch source footage into (backward + forward). A
# script needing more coverage than this has no real footage to cover it,
# no matter how the clips are picked.
MAX_SCRIPT_EXTRA_SECONDS = 20
# How far (seconds) a clip's source_start/source_end may sit from the nearest
# real transcript segment boundary and still be trusted/snapped for a clean cut.
CLIP_BOUNDARY_TOLERANCE = 4.0

# Cold-open teaser (hook_clip) length guards: repaired rather than rejected,
# since a botched teaser pick should degrade to "no cold open", never fail
# the whole plan. Too-short picks are grown backward (buildup before the
# peak) so the payoff still lands at the teaser's end.
HOOK_CLIP_TARGET_SECONDS = 4.0
HOOK_CLIP_MIN_SECONDS = 3.0
HOOK_CLIP_MAX_SECONDS = 6.0
HOOK_CLIP_TRIM_SECONDS = 5.0

# Hook text: overlaid on screen for ~3.5s - past this many words it can't be
# read before it fades, and long hooks read as essay, not scroll-stopper.
HOOK_MAX_WORDS = 12

# Raw mode: the whole cut clip plays from its start right after the teaser,
# so a teaser moment inside the clip's opening stretch would play twice
# back-to-back - require the teased moment to sit at least this deep in.
RAW_TEASER_MIN_START_SECONDS = 15.0

ALLOWED_TRANSITIONS = {
    "hard_cut", "fade", "dissolve", "wipeleft", "wiperight",
    "slideup", "slidedown", "circleopen", "circleclose", "pixelize",
}
SYSTEM_PROMPT = f"""You are writing a render plan for a short vertical (9:16) commentary/reaction video.
You will be given a transcript of a source video with segment timestamps. Some segments are tagged [LOUD REACTION] - the streamer's voice spiked well above their normal level there (an actual excited reaction). That's a bonus signal worth favoring when choosing between otherwise-similar moments, but most good clips won't have one - don't require it.
Produce a JSON object with exactly these fields:
- title: short internal title for the draft
- hooks: a list of exactly 5 distinct hook options, ORDERED STRONGEST FIRST - each ONE punchy sentence of AT MOST 12 words, overlaid on screen as the video opens. Clip-farming clickbait energy: overclaim a little, bait curiosity, pick a side, aim straight at the viewer ("you") - the kind of line that stops a scroll. Never a neutral description. Vary the angle across the 5: shock, a question or challenge, a bold claim, direct address, an absurd specific detail.
- hook_clip: {{"source_start": number, "source_end": number}} - the single most explosive 3-5 second moment in the transcript (the scream, the punchline, the moment of impact, plus a beat of buildup before it). It plays as a cold-open teaser before the video starts from the beginning - the viewer sees the payoff first, then stays to watch how it got there. Put the peak near the END of the teaser so it climaxes right before the cut; unlike clips below it does NOT need to land on segment boundaries.
  CRITICAL: the teased moment must fall inside one of your clips' ranges, but NEVER the FIRST clip - the video has to build up to the payoff, so the moment must resurface in a later clip (ideally past the midpoint of the sequence). If the teaser were followed immediately by the same footage again, the opening feels broken - structure your clips so earlier ones are buildup/context and the payoff lands late.
- script: the full voiceover narration script, written to be read aloud. It must be substantial: at least {MIN_SCRIPT_WORDS} words, so it runs at least 60 seconds at a natural speaking pace. Do not pad with filler or repetition - add genuine additional reaction, context, or opinion to reach the length.
  Voice: you're a hyped-up friend reacting live over the viewer's shoulder, not a critic writing a review. Talk TO the viewer ("you," "watch this," "I can't believe he just—") instead of describing the clip in the third person. Have an actual opinion or hot take on what's happening, not a neutral recap of events. Mix short, punchy sentences with the occasional longer one - vary the rhythm, don't lock into a uniform list-like cadence.
  Lean into clickbait/engagement-bait energy throughout, not just in the hook: exaggerate for effect, pick a fight, take a strong or slightly controversial stance on what's happening (the gameplay, the player's decisions, the situation) rather than staying neutral. Provoke a little - a take someone might want to argue with in the comments is good. Stay focused on the video's own content though, not real-world hot-button issues (politics, religion, etc.) - the controversy should live inside the video's subject matter, not anything that could actually hurt someone.
  Banned: never use the construction "isn't just X, it's Y" (or close variants) or other essay-review crutches like "a testament to," "masterclass in," "paradigm shift," "at its core," or "in the world of X." Avoid stiff/academic vocabulary ("palpable," "schadenfreude," "meta-narrative"). If a line sounds like it belongs in a film review, cut it.
  End on a punchline, a question, or a cliffhanger that makes someone want to keep watching - not a tidy summarizing conclusion.
- clips: a list of 2-6 objects, each with:
  - source_start: number (seconds) - MUST exactly match the start of one of the transcript segments below, so the cut lands cleanly at the start of a sentence (the original audio plays underneath your narration, so a mid-sentence cut is audible and sounds broken).
  - source_end: number (seconds) - MUST exactly match the end of one of the transcript segments below (the same one or a later one), for the same reason.
  - reason: short string explaining why this moment was picked
  - transition_in (required): how this clip transitions in from the previous one. One of {sorted(ALLOWED_TRANSITIONS)}. Use "hard_cut" for the first clip. For the rest, favor punchy transitions ("fade", "dissolve", "wipeleft", "wiperight", "slideup", "slidedown", "circleopen", "circleclose", "pixelize") over "hard_cut" - only use "hard_cut" when a hard cut is genuinely the stronger choice (e.g. a jarring reveal).
  - zoom (strongly encouraged): {{"start_scale": number, "end_scale": number}} for a Ken Burns zoom over the clip - go bold (e.g. 1.0 -> 1.4, or 1.2 -> 1.0 for a pull-back). Give most clips a zoom; only omit it on a clip that needs to sit completely still.
This is a fast-paced, high-energy edit - be bold with zoom and transitions rather than conservative.
All clip timestamps must fall within the transcript's total duration and source_end must be greater than source_start.
Respond with ONLY the JSON object, no other text."""


class Zoom(BaseModel):
    start_scale: float
    end_scale: float


class Clip(BaseModel):
    source_start: float
    source_end: float
    reason: str
    transition_in: str = "hard_cut"
    zoom: Zoom | None = None


class HookClip(BaseModel):
    source_start: float
    source_end: float


class RenderPlan(BaseModel):
    title: str
    hooks: list[str] = []
    hook: str | None = None  # resolved from hooks during validation (kept for downstream/back-compat)
    script: str
    clips: list[Clip]
    hook_clip: HookClip | None = None


class RawPlan(BaseModel):
    """Raw mode: the clip plays with the streamer's own audio - no script,
    no source-clip picking. The LLM only supplies packaging: title, hook
    options, and the cold-open teaser pick."""
    title: str
    hooks: list[str] = []
    hook: str | None = None
    hook_clip: HookClip | None = None


def resolve_hook(hooks: list[str], fallback: str | None) -> str:
    """Picks the strongest usable hook: the model orders candidates
    strongest-first, so take the first one short enough to be read before
    the overlay fades; if none fit, the shortest candidate."""
    candidates = [h.strip() for h in hooks if h and h.strip()]
    if fallback and fallback.strip():
        candidates.append(fallback.strip())
    if not candidates:
        raise ValueError("No hook provided - supply a 'hooks' list of 5 options (each at most 12 words).")
    fitting = [h for h in candidates if len(h.split()) <= HOOK_MAX_WORDS]
    return fitting[0] if fitting else min(candidates, key=lambda h: len(h.split()))


def load_transcript(record_id: str) -> dict:
    path = TRANSCRIPTS_DIR / f"{record_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_transcript_for_prompt(transcript: dict) -> str:
    lines = [f"Total duration: {transcript['duration']:.1f}s\n"]
    for seg in transcript["segments"]:
        marker = " [LOUD REACTION]" if seg.get("loud") else ""
        lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}]{marker} {seg['text']}")
    return "\n".join(lines)


def validate_plan(raw_json: dict, duration: float, segments: list[dict]) -> RenderPlan:
    plan = RenderPlan.model_validate(raw_json)
    plan.hook = resolve_hook(plan.hooks, plan.hook)

    word_count = len(plan.script.split())
    if word_count < MIN_SCRIPT_WORDS:
        raise ValueError(
            f"Script is only {word_count} words - needs at least {MIN_SCRIPT_WORDS} "
            f"words to run ~60 seconds when read aloud. Expand it with genuine additional commentary."
        )
    max_words = int((duration + MAX_SCRIPT_EXTRA_SECONDS) / 60 * WORDS_PER_MINUTE)
    if word_count > max_words:
        # For short clips the valid word window is narrow and the local model
        # chronically overshoots by a handful of words - trim trailing
        # sentences to fit instead of burning retries (a slightly shorter
        # ending beats a failed plan). Only a gross overshoot that trimming
        # can't fix still raises for a corrective retry.
        sentences = re.split(r"(?<=[.!?])\s+", plan.script.strip())
        while sentences and len(" ".join(sentences).split()) > max_words:
            sentences.pop()
        trimmed = " ".join(sentences)
        if len(trimmed.split()) >= MIN_SCRIPT_WORDS:
            plan.script = trimmed
        else:
            raise ValueError(
                f"Script is {word_count} words - that runs longer than this {duration:.0f}s clip has footage "
                f"to cover (max ~{max_words} words). Trim it down closer to the clip's own length."
            )

    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    for clip in plan.clips:
        if clip.source_start < 0 or clip.source_end > duration:
            raise ValueError(
                f"Clip [{clip.source_start}, {clip.source_end}] falls outside "
                f"transcript duration 0-{duration:.1f}"
            )
        if clip.source_end <= clip.source_start:
            raise ValueError(
                f"Clip source_end ({clip.source_end}) must be greater than "
                f"source_start ({clip.source_start})"
            )
        if clip.transition_in not in ALLOWED_TRANSITIONS:
            raise ValueError(
                f"Clip transition_in '{clip.transition_in}' is not one of {sorted(ALLOWED_TRANSITIONS)}"
            )
        start_dist = min(abs(s - clip.source_start) for s in starts)
        end_dist = min(abs(e - clip.source_end) for e in ends)
        if start_dist > CLIP_BOUNDARY_TOLERANCE or end_dist > CLIP_BOUNDARY_TOLERANCE:
            raise ValueError(
                f"Clip source_start {clip.source_start} (nearest real segment boundary is {start_dist:.1f}s "
                f"away) and/or source_end {clip.source_end} (nearest is {end_dist:.1f}s away) don't match an "
                f"actual segment boundary from the transcript - copy timestamps directly from it, don't invent them."
            )
        clip.source_start = min(starts, key=lambda s: abs(s - clip.source_start))
        clip.source_end = min(ends, key=lambda e: abs(e - clip.source_end))

    if plan.hook_clip is not None:
        hc = plan.hook_clip
        hc.source_start = max(hc.source_start, 0.0)
        hc.source_end = min(hc.source_end, duration)
        if hc.source_end - hc.source_start > HOOK_CLIP_MAX_SECONDS:
            hc.source_end = hc.source_start + HOOK_CLIP_TRIM_SECONDS
        if hc.source_end - hc.source_start < HOOK_CLIP_MIN_SECONDS:
            # Grow backward first (buildup, keeping the payoff at the end),
            # then forward if the clip starts too close to 0.
            hc.source_start = max(hc.source_end - HOOK_CLIP_TARGET_SECONDS, 0.0)
            hc.source_end = min(max(hc.source_start + HOOK_CLIP_TARGET_SECONDS, hc.source_end), duration)
        if hc.source_end - hc.source_start < 1.0:
            plan.hook_clip = None

    if plan.hook_clip is not None:
        # The teased payoff must resurface later in the video. If the moment
        # never appears in a picked clip, or sits in the FIRST clip (teaser
        # followed immediately by the same footage again), the cold open
        # hurts more than it helps - drop it rather than fail the plan.
        hc = plan.hook_clip
        mid = (hc.source_start + hc.source_end) / 2
        container = next(
            (i for i, c in enumerate(plan.clips) if c.source_start <= mid <= c.source_end),
            None,
        )
        if container is None or container == 0:
            plan.hook_clip = None

    return plan


def generate_render_plan(record_id: str) -> RenderPlan:
    config = load_config()["llm"]
    transcript = load_transcript(record_id)
    flag_loud_segments(Path(transcript["audio_path"]), transcript["segments"])
    transcript_text = format_transcript_for_prompt(transcript)

    target_words = int(transcript["duration"] / 60 * WORDS_PER_MINUTE)
    messages = [
        {"role": "system", "content": "/no_think\n" + SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Transcript ({transcript['duration']:.0f}s total):\n{transcript_text}\n\n"
                f"This clip is only {transcript['duration']:.0f}s long. Target a script around {target_words} "
                f"words (~{MIN_SCRIPT_WORDS} minimum) - it needs to stay close to the clip's own length, since "
                f"the picked clips can only draw footage from within this {transcript['duration']:.0f}s file."
            ),
        },
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
            raw_json = json.loads(content)
            plan = validate_plan(raw_json, transcript["duration"], transcript["segments"])
            # Teaser priority: the VOD-level measured pick from split_movie,
            # else blend the LLM's pick with this clip's audio peaks. Either
            # way the payoff must resurface in a non-first plan clip.
            later_clip_ranges = [(c.source_start, c.source_end) for c in plan.clips[1:]]
            handed_down = ingested_teaser(record_id)
            if handed_down is not None and any(
                lo <= (handed_down.source_start + handed_down.source_end) / 2 <= hi for lo, hi in later_clip_ranges
            ):
                plan.hook_clip = handed_down
            elif later_clip_ranges:
                plan.hook_clip = choose_teaser(plan.hook_clip, record_id, transcript["duration"], later_clip_ranges)
            return plan
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That response was invalid: {e}\nPlease fix it and respond with ONLY the corrected JSON object.",
            })

    raise RuntimeError(f"LLM failed to produce a valid render plan after {MAX_ATTEMPTS} attempts: {last_error}")


def plan_and_save(record_id: str) -> Path:
    plan = generate_render_plan(record_id)

    RENDER_PLANS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    plan_path = RENDER_PLANS_DIR / f"{record_id}.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, indent=2)

    script_path = SCRIPTS_DIR / f"{record_id}.txt"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(plan.script)

    return plan_path


RAW_SYSTEM_PROMPT = f"""You are packaging a raw streamer clip for TikTok/Shorts. The clip plays as-is with the streamer's own voice and game audio - you are NOT writing a script or narration. Your only job is the packaging around it.
You will be given a timestamped transcript of the clip. Some segments are tagged [LOUD REACTION] - the streamer's voice spiked well above normal there.
Produce a JSON object with exactly these fields:
- title: short internal title for the draft
- hooks: a list of exactly 5 distinct hook options, ORDERED STRONGEST FIRST - each ONE punchy sentence of AT MOST 12 words, overlaid on screen as the video opens. Clip-farming clickbait energy: overclaim a little, bait curiosity, pick a side, aim straight at the viewer ("you"). Never a neutral description. Vary the angle across the 5: shock, a question or challenge, a bold claim, direct address, an absurd specific detail.
- hook_clip: {{"source_start": number, "source_end": number}} - the single most explosive 3-5 second moment (the scream, the punchline, the impact, plus a beat of buildup), played as a cold-open teaser before the clip runs from its start. Put the peak near the END of the teaser. CRITICAL: it must start at least {RAW_TEASER_MIN_START_SECONDS:.0f} seconds into the clip - the clip plays from the beginning right after the teaser, so teasing the opening would show the same footage twice back-to-back.
Respond with ONLY the JSON object, no other text."""


def validate_raw_plan(raw_json: dict, duration: float) -> RawPlan:
    plan = RawPlan.model_validate(raw_json)
    plan.hook = resolve_hook(plan.hooks, plan.hook)

    if plan.hook_clip is not None:
        hc = plan.hook_clip
        hc.source_start = max(hc.source_start, 0.0)
        hc.source_end = min(hc.source_end, duration)
        if hc.source_end - hc.source_start > HOOK_CLIP_MAX_SECONDS:
            hc.source_end = hc.source_start + HOOK_CLIP_TRIM_SECONDS
        if hc.source_end - hc.source_start < HOOK_CLIP_MIN_SECONDS:
            hc.source_start = max(hc.source_end - HOOK_CLIP_TARGET_SECONDS, 0.0)
            hc.source_end = min(max(hc.source_start + HOOK_CLIP_TARGET_SECONDS, hc.source_end), duration)
        # Teasing the clip's own opening plays the same footage twice - drop.
        if hc.source_start < RAW_TEASER_MIN_START_SECONDS or hc.source_end - hc.source_start < 1.0:
            plan.hook_clip = None

    return plan


def ingested_teaser(record_id: str) -> HookClip | None:
    """The teaser split_movie measured at the VOD level and threaded through
    ingest - the most trusted source, since it was chosen with chat + audio
    signals the per-clip stages can't see. None for manually staged clips."""
    try:
        teaser = load_ingest_record(record_id).get("teaser")
    except FileNotFoundError:
        return None
    if not teaser:
        return None
    return HookClip(source_start=teaser["start"], source_end=teaser["end"])


def pick_peak_teaser(record_id: str, duration: float, allowed_ranges: list[tuple[float, float]] | None = None) -> HookClip | None:
    """Deterministic cold-open pick: the clip's loudest audio moment (RMS
    peak, i.e. the scream/laugh/impact) IS the teaser - measured, not
    guessed, after LLM teaser picks kept being weak or landing in guarded
    stretches. The peak sits near the teaser's end so it climaxes right
    before the cut.

    allowed_ranges restricts where the peak may come from: narrated mode
    passes the non-first clips' source ranges (the teased payoff must
    resurface later in the video); raw mode omits it and gets the default
    guard - anywhere past the opening stretch the clip replays right after."""
    audio_path = TRANSCRIPTS_DIR / f"{record_id}.wav"
    if not audio_path.exists():
        return None

    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    win = int(sample_rate * 0.5)
    if samples.size < win:
        return None
    n = len(samples) // win
    rms = np.sqrt(np.mean(samples[: n * win].reshape(n, win) ** 2, axis=1))

    if allowed_ranges is None:
        allowed_ranges = [(RAW_TEASER_MIN_START_SECONDS + 3.0, duration)]

    centers = (np.arange(n) + 0.5) * 0.5
    allowed = np.zeros(n, dtype=bool)
    for lo, hi in allowed_ranges:
        # The clip must still have somewhere to go after the peak.
        allowed |= (centers >= lo) & (centers <= min(hi, duration - 1.5))
    if not allowed.any():
        return None

    peak_idx = int(np.argmax(np.where(allowed, rms, -1.0)))
    peak_t = float(centers[peak_idx])

    start = max(peak_t - 3.0, 0.0)
    end = min(peak_t + 1.5, duration)
    if end - start < 1.0:
        return None
    return HookClip(source_start=start, source_end=end)


def choose_teaser(
    llm_pick: HookClip | None, record_id: str, duration: float,
    allowed_ranges: list[tuple[float, float]] | None = None,
) -> HookClip | None:
    """Blends the two teaser signals: the LLM understands MEANING (punchline,
    story beat) but proved unreliable alone; audio energy is measured truth
    but only captures loud moments. Policy: keep the LLM's (already
    guard-validated) pick when the audio backs it - a top peak lands in/near
    it, or it at least has above-median energy (the calm-but-great story
    beat). Otherwise use the loudest measured peak. A clip with no usable
    audio profile keeps whatever the LLM gave."""
    profile = _load_rms_profile(record_id)
    if profile is None:
        return llm_pick
    centers, rms = profile

    audio_pick = pick_peak_teaser(record_id, duration, allowed_ranges)
    if llm_pick is None:
        return audio_pick
    if audio_pick is None:
        return llm_pick

    in_pick = (centers >= llm_pick.source_start) & (centers <= llm_pick.source_end)
    near_pick = (centers >= llm_pick.source_start - 2.0) & (centers <= llm_pick.source_end + 2.0)
    if not in_pick.any():
        return audio_pick

    top5 = np.argsort(rms)[-5:]
    if near_pick[top5].any():
        return llm_pick
    if float(np.max(rms[in_pick])) >= float(np.median(rms)):
        return llm_pick
    return audio_pick


def _load_rms_profile(record_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(window_center_seconds, rms) in 0.5s steps for the record's transcribed
    audio, or None when the wav is missing/too short."""
    audio_path = TRANSCRIPTS_DIR / f"{record_id}.wav"
    if not audio_path.exists():
        return None
    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    win = int(sample_rate * 0.5)
    if win <= 0 or samples.size < win:
        return None
    n = len(samples) // win
    rms = np.sqrt(np.mean(samples[: n * win].reshape(n, win) ** 2, axis=1))
    return (np.arange(n) + 0.5) * 0.5, rms


def generate_raw_plan(record_id: str) -> RawPlan:
    config = load_config()["llm"]
    transcript = load_transcript(record_id)
    flag_loud_segments(Path(transcript["audio_path"]), transcript["segments"])
    transcript_text = format_transcript_for_prompt(transcript)

    messages = [
        {"role": "system", "content": "/no_think\n" + RAW_SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript ({transcript['duration']:.0f}s total):\n{transcript_text}"},
    ]

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = call_llm(messages, config)
        except requests.exceptions.RequestException as e:
            last_error = e
            continue
        try:
            raw_json = json.loads(content)
            plan = validate_raw_plan(raw_json, transcript["duration"])
            # Teaser priority: the VOD-level measured pick from split_movie,
            # else blend the LLM's pick with this clip's audio peaks.
            handed_down = ingested_teaser(record_id)
            if handed_down is not None:
                plan.hook_clip = handed_down
            else:
                plan.hook_clip = choose_teaser(plan.hook_clip, record_id, transcript["duration"])
            return plan
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That response was invalid: {e}\nPlease fix it and respond with ONLY the corrected JSON object.",
            })

    raise RuntimeError(f"LLM failed to produce a valid raw plan after {MAX_ATTEMPTS} attempts: {last_error}")


def plan_raw_and_save(record_id: str) -> Path:
    plan = generate_raw_plan(record_id)

    RENDER_PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = RENDER_PLANS_DIR / f"{record_id}.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({**plan.model_dump(), "mode": "raw"}, f, indent=2)

    return plan_path


def main():
    parser = argparse.ArgumentParser(description="Generate an LLM render plan from a transcript.")
    parser.add_argument("--id", required=True, help="Transcript record id from transcribe.py.")
    args = parser.parse_args()

    print(plan_and_save(args.id))


if __name__ == "__main__":
    main()
