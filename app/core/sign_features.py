"""
sign_features.py  —  SHARED feature extraction for Sign->Text.

  *** PARITY CONTRACT ***
  This exact file is used BOTH in training (Colab notebook) and in the live app.
  If you change anything here, you MUST retrain and ship the same file to the app.
  Do not "improve" one side only — that is what broke accuracy before.

Pipeline (identical on both sides):
  1. Read video / camera frames as RGB, full resolution.
  2. Run MediaPipe HandLandmarker in IMAGE mode (frame-independent -> sampling
     density doesn't matter, so train and live always agree).
  3. Order hands deterministically by handedness: slot 0 = Left, slot 1 = Right.
     Missing hand -> that 63-float block stays zero.
  4. Uniformly sample SEQ_LEN frames across the whole single-gloss clip/segment.
  5. Normalize the WHOLE sequence: subtract the clip's mean hand anchor and divide
     by mean hand size -> position- and scale-invariant, motion preserved.
"""

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    _MP_OK = True
except Exception:
    _MP_OK = False

SEQ_LEN = 30
HANDS = 2
LM = 21
C = 3
FEATURES = HANDS * LM * C  # 126

WRIST = 0
MIDDLE_MCP = 9


# ---------------------------------------------------------------------------
# MediaPipe landmarker (IMAGE mode = frame-independent = train/live parity)
# ---------------------------------------------------------------------------
def make_landmarker(task_path: str, mode: str = "image", result_callback=None):
    m = str(mode).lower()
    if m == "video":
        rm = vision.RunningMode.VIDEO
    elif m in ("live_stream", "live", "stream"):
        rm = vision.RunningMode.LIVE_STREAM
    else:
        rm = vision.RunningMode.IMAGE

    base = mp_python.BaseOptions(model_asset_path=task_path)
    kwargs = dict(
        base_options=base,
        running_mode=rm,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    if rm == vision.RunningMode.LIVE_STREAM:
        kwargs["result_callback"] = result_callback
    opts = vision.HandLandmarkerOptions(**kwargs)
    return vision.HandLandmarker.create_from_options(opts)


def detect_image(landmarker, rgb_frame):
    """IMAGE mode: synchronous, returns result."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect(mp_image)


def detect_video(landmarker, rgb_frame, ts_ms):
    """VIDEO mode: synchronous w/ tracking. ts_ms must strictly increase."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, int(ts_ms))


def detect_async(landmarker, rgb_frame, ts_ms):
    """LIVE_STREAM mode: non-blocking; result arrives via result_callback.
    ts_ms must strictly increase."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    landmarker.detect_async(mp_image, int(ts_ms))


# ---------------------------------------------------------------------------
# Result -> (kp[126], present[2])  ordered by handedness
# ---------------------------------------------------------------------------
def landmarks_from_result(result):
    kp = np.zeros(FEATURES, dtype=np.float32)
    present = np.zeros(HANDS, dtype=np.float32)
    hands = getattr(result, "hand_landmarks", None)
    if not hands:
        return kp, present

    n = min(2, len(hands))
    slot = [None, None]  # [Left, Right]
    handed = getattr(result, "handedness", None)
    for i in range(n):
        cat = None
        try:
            cat = handed[i][0].category_name
        except Exception:
            cat = None
        if cat == "Left" and slot[0] is None:
            slot[0] = i
        elif cat == "Right" and slot[1] is None:
            slot[1] = i

    used = {x for x in slot if x is not None}
    remaining = [i for i in range(n) if i not in used]
    for s in range(2):
        if slot[s] is None and remaining:
            slot[s] = remaining.pop(0)

    for s in range(2):
        i = slot[s]
        if i is None:
            continue
        present[s] = 1.0
        for j, lm in enumerate(hands[i]):
            base = s * 63 + j * 3
            kp[base:base + 3] = (lm.x, lm.y, lm.z)
    return kp, present


# ---------------------------------------------------------------------------
# Sequence normalization (position + scale invariant, motion preserved)
# ---------------------------------------------------------------------------
def normalize_sequence(seq, present):
    """seq: (T, 126) raw handedness-ordered. present: (T, 2). Returns (T, 126)."""
    seq = np.asarray(seq, dtype=np.float32).copy()
    T = seq.shape[0]
    pts = seq.reshape(T, HANDS, LM, C)
    pres = np.asarray(present, dtype=np.float32).reshape(T, HANDS) > 0.5

    anchors = []
    sizes = []
    for t in range(T):
        ws, ss = [], []
        for h in range(HANDS):
            if pres[t, h]:
                wrist = pts[t, h, WRIST]
                mcp = pts[t, h, MIDDLE_MCP]
                ws.append(wrist)
                ss.append(np.linalg.norm(mcp - wrist))
        if ws:
            anchors.append(np.mean(ws, axis=0))
            sizes.append(np.mean(ss))

    if not anchors:
        return np.zeros((T, FEATURES), dtype=np.float32)

    origin = np.mean(anchors, axis=0)
    scale = float(np.mean(sizes))
    if scale < 1e-6:
        scale = 1e-6

    out = np.zeros_like(pts)
    for t in range(T):
        for h in range(HANDS):
            if pres[t, h]:
                out[t, h] = (pts[t, h] - origin) / scale
    return out.reshape(T, FEATURES).astype(np.float32)


def sample_indices(n_frames: int, k: int = SEQ_LEN):
    return np.linspace(0, max(n_frames - 1, 0), k).astype(int)


# ---------------------------------------------------------------------------
# Whole-video -> normalized features (used in training)
# ---------------------------------------------------------------------------
def raw_sequence_from_video(video_path: str, task_path: str, mode: str = "video"):
    """Extract (seq[SEQ_LEN,126], present[SEQ_LEN,2]) from a single-gloss clip.

    mode="video" (default, recommended): a FRESH landmarker instance per video,
        running in VIDEO mode. Every frame is processed in order with temporal
        tracking, then SEQ_LEN frames are uniformly sampled. This matches the live
        app, which also tracks every frame and samples afterward.
    mode="image": legacy. Sample SEQ_LEN frame indices first, detect each
        independently (no tracking).

    Returns (None, None) if the clip has no frames. The instance is created and
    closed inside this call, so tracking state never leaks between videos.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    if not frames:
        return None, None

    if str(mode).lower() == "video":
        landmarker = make_landmarker(task_path, mode="video")  # fresh instance per video
        per_kp, per_pres = [], []
        last_ts = 0
        for i, fr in enumerate(frames):
            ts = int(round(i * 1000.0 / max(fps, 1.0)))
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            res = detect_video(landmarker, fr, ts)   # tracked, in order
            kp, pr = landmarks_from_result(res)
            per_kp.append(kp)
            per_pres.append(pr)
        landmarker.close()
        idxs = sample_indices(len(frames))
        seq = np.stack(per_kp)[idxs].astype(np.float32)
        pres = np.stack(per_pres)[idxs].astype(np.float32)
        return seq, pres

    # legacy IMAGE mode
    landmarker = make_landmarker(task_path, mode="image")
    idxs = sample_indices(len(frames))
    seq = np.zeros((SEQ_LEN, FEATURES), dtype=np.float32)
    pres = np.zeros((SEQ_LEN, HANDS), dtype=np.float32)
    for r, ix in enumerate(idxs):
        res = detect_image(landmarker, frames[ix])
        kp, pr = landmarks_from_result(res)
        seq[r] = kp
        pres[r] = pr
    landmarker.close()
    return seq, pres


def features_from_video(video_path: str, task_path: str):
    """Full pipeline: video -> normalized (SEQ_LEN, 126). None if no frames."""
    seq, pres = raw_sequence_from_video(video_path, task_path)
    if seq is None:
        return None
    return normalize_sequence(seq, pres)
