"""Chat-replay spike detection for stream VOD highlight selection.

Parses a chat replay file (yt-dlp's `.live_chat.json` JSONL format, or the
`chat-downloader` package's JSON array format) and tags transcript segments
where the chat message rate erupted - thousands of viewers reacting is the
strongest available signal that a moment is clip-worthy, and it catches
visual moments the transcript alone can't see.
"""
import json
import re
from pathlib import Path

import numpy as np

BUCKET_SECONDS = 10.0
# Chat reacts a few seconds after the moment that caused it, so a spiking
# bucket's tag is spread this far backward onto the segments that caused it.
CHAT_REACTION_LAG = 15.0
# A bucket spikes when its weighted message count clears both the relative
# ratio over the stream's baseline and an absolute floor (the floor guards
# small chats, where 2 messages against a baseline of 1 is noise, not hype).
SPIKE_RATIO = 2.0
MIN_SPIKE_WEIGHT = 4.0
# Messages that read as a reaction (emote spam, "NO WAY", "clip it") count
# extra toward the bucket weight versus ordinary chatter.
HYPE_WEIGHT = 2.0

HYPE_PATTERN = re.compile(
    r"💀|😂|🤣|😭|\blm(f?)ao+\b|\blo+l\b|\bomg+\b|\bno way\b|\bwtf\b|\bclip (it|that)\b"
    r"|\binsane\b|\bcrazy\b|\bnah+\b|^w+$|^l+$",
    re.IGNORECASE,
)


def _runs_to_text(message: dict) -> str:
    parts = []
    for run in message.get("runs", []):
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            emoji = run["emoji"]
            shortcuts = emoji.get("shortcuts") or []
            parts.append(shortcuts[0] if shortcuts else emoji.get("emojiId", ""))
    return "".join(parts)


def _parse_ytdlp_line(line: str) -> tuple[float, str] | None:
    entry = json.loads(line)
    replay = entry.get("replayChatItemAction")
    if not replay:
        return None
    offset_msec = replay.get("videoOffsetTimeMsec", entry.get("videoOffsetTimeMsec"))
    if offset_msec is None:
        return None
    for action in replay.get("actions", []):
        item = action.get("addChatItemAction", {}).get("item", {})
        renderer = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if renderer:
            return float(offset_msec) / 1000.0, _runs_to_text(renderer.get("message", {}))
    return None


def load_chat_messages(path: Path) -> list[tuple[float, str]]:
    """Returns [(video_offset_seconds, message_text), ...] from either
    supported chat replay format, skipping pre-stream (negative offset) and
    malformed entries."""
    text = path.read_text(encoding="utf-8").strip()
    messages: list[tuple[float, str]] = []

    if text.startswith("["):  # chat-downloader JSON array
        for entry in json.loads(text):
            t = entry.get("time_in_seconds")
            msg = entry.get("message")
            if t is not None and t >= 0 and msg:
                messages.append((float(t), str(msg)))
        return messages

    for line in text.splitlines():  # yt-dlp live_chat.json (JSONL)
        line = line.strip()
        if not line:
            continue
        try:
            parsed = _parse_ytdlp_line(line)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if parsed and parsed[0] >= 0:
            messages.append(parsed)
    return messages


def flag_chat_spike_segments(messages: list[tuple[float, str]], segments: list[dict]) -> tuple[int, list[dict]]:
    """Marks segment["chat_spike"] = True on every transcript segment that
    overlaps a chat-eruption window (including the reaction lag spread
    backward onto the moment that caused it). Mutates segments in place and
    returns (flagged_count, spike_events) - each spike event is
    {"time": approximate moment that caused the eruption, "weight": bucket
    weight}, usable as measured ground-truth highlight anchors independent
    of any LLM judgment."""
    if not messages or not segments:
        return 0, []

    end_time = max(max(t for t, _ in messages), max(s["end"] for s in segments))
    n_buckets = int(end_time // BUCKET_SECONDS) + 1
    weights = np.zeros(n_buckets)
    for t, msg in messages:
        b = int(t // BUCKET_SECONDS)
        if b < n_buckets:
            weights[b] += HYPE_WEIGHT if HYPE_PATTERN.search(msg.strip()) else 1.0

    nonzero = weights[weights > 0]
    if nonzero.size == 0:
        return 0, []
    # 75th percentile of active buckets, so the baseline tracks what "normal
    # chatter" looks like for this stream's chat speed - a spike must stand
    # out relative to that, not against near-empty lulls.
    baseline = float(np.percentile(nonzero, 75))
    threshold = max(baseline * SPIKE_RATIO, MIN_SPIKE_WEIGHT)

    spike_buckets = [b for b in range(n_buckets) if weights[b] >= threshold]
    spike_windows = [(b * BUCKET_SECONDS - CHAT_REACTION_LAG, (b + 1) * BUCKET_SECONDS) for b in spike_buckets]
    spike_events = [
        {"time": max(b * BUCKET_SECONDS - CHAT_REACTION_LAG / 2, 0.0), "weight": float(weights[b])}
        for b in spike_buckets
    ]
    if not spike_windows:
        return 0, []

    flagged = 0
    for seg in segments:
        if any(seg["start"] < w_end and seg["end"] > w_start for w_start, w_end in spike_windows):
            seg["chat_spike"] = True
            flagged += 1
    return flagged, spike_events
