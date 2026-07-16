from pathlib import Path

import cv2

SAMPLE_FRAMES = 24
MIN_DETECTION_RATE = 0.3   # fraction of sampled frames a region must appear in to be trusted
CLUSTER_TOLERANCE = 0.06   # fraction of frame dimension boxes may drift and still count as "the same" region
MAX_FACE_AREA_RATIO = 0.2  # reject detections implausibly large to be a small webcam bubble
CAM_PADDING = 0.2          # expand the detected face box by this fraction per side to capture the full cam bubble


def sample_frames(video_path: Path, count: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frames = []
    for i in range(count):
        frame_idx = int(total_frames * (i + 0.5) / count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def detect_cam_region(video_path: Path) -> dict | None:
    """Auto-detects a static webcam overlay by running face detection across
    sampled frames and looking for a box that recurs in roughly the same
    spot - a face in the actual game footage moves/appears inconsistently,
    a webcam overlay sits in the same corner the whole video. Returns a
    ratio-based region {x_ratio, y_ratio, w_ratio, h_ratio} (0-1, relative to
    frame size) or None if no consistent cam is found."""
    frames = sample_frames(video_path, SAMPLE_FRAMES)
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    boxes = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40)):
            if (w * h) / (width * height) > MAX_FACE_AREA_RATIO:
                continue
            boxes.append((x / width, y / height, w / width, h / height))

    if not boxes:
        return None

    clusters: list[list[tuple[float, float, float, float]]] = []
    for box in boxes:
        bx, by = box[0], box[1]
        match = next(
            (c for c in clusters if abs(c[0][0] - bx) < CLUSTER_TOLERANCE and abs(c[0][1] - by) < CLUSTER_TOLERANCE),
            None,
        )
        if match is not None:
            match.append(box)
        else:
            clusters.append([box])

    best = max(clusters, key=len)
    if len(best) / len(frames) < MIN_DETECTION_RATE:
        return None

    x_ratio = sum(b[0] for b in best) / len(best)
    y_ratio = sum(b[1] for b in best) / len(best)
    w_ratio = sum(b[2] for b in best) / len(best)
    h_ratio = sum(b[3] for b in best) / len(best)

    x_ratio = max(x_ratio - w_ratio * CAM_PADDING, 0)
    y_ratio = max(y_ratio - h_ratio * CAM_PADDING, 0)
    w_ratio = min(w_ratio * (1 + 2 * CAM_PADDING), 1 - x_ratio)
    h_ratio = min(h_ratio * (1 + 2 * CAM_PADDING), 1 - y_ratio)

    return {
        "x_ratio": float(x_ratio),
        "y_ratio": float(y_ratio),
        "w_ratio": float(w_ratio),
        "h_ratio": float(h_ratio),
    }
