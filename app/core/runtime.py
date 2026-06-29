# app/core/runtime.py
import os
import sys
import json
import time
import threading
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from app.core import sign_features as sf

try:
    import requests
except Exception:
    requests = None

# =========================
# Debug
# =========================
VERBOSE = True
def vlog(*args):
    if VERBOSE:
        print("[CSLR]", *args, flush=True)

# =========================
# Files
# =========================
MODEL_PATH = "model.keras"
TASK_PATH = "hand_landmarker.task"
LABEL_MAP_PATH = "label_map.json"
GLOSS_DB_PATH = "gloss_db.txt"
CONFIG_PATH = "config.json"

# =========================
# Config
# =========================
def _load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

CFG = _load_config(CONFIG_PATH)

SEQ_LEN = sf.SEQ_LEN
FEATURES = sf.FEATURES

TAKE_SECONDS = float(CFG.get("take_seconds", 2.0))
TAKE_MS = int(TAKE_SECONDS * 1000)

CAM_INDEX = int(CFG.get("cam_index", 0))
CAM_W, CAM_H = 640, 480
CAM_FPS_REQ = 30
CAM_BACKEND = str(CFG.get("cam_backend", "auto")).lower()

PREVIEW_W, PREVIEW_H = 640, 480
PREVIEW_FPS = 12

RING_SECONDS = 8
RING_MAX = int(RING_SECONDS * CAM_FPS_REQ * 3)

# Feature normalization MUST match the trained model.
#   "sequence" -> sign_features.normalize_sequence  (new model)
#   "none"     -> raw landmarks                      (legacy model)
FEATURE_NORMALIZE = str(CFG.get("feature_normalize", "sequence")).lower()

# MediaPipe running mode for the LIVE app:
#   "video" -> tracking between frames = much faster = more frames per sign (recommended)
#   "image" -> full detection every frame (matches training exactly, but slow on CPU)
# Training/test always use IMAGE mode (sparse sampling); small live/train landmark
# differences are absorbed by normalization + jitter augmentation.
MP_MODE = str(CFG.get("mp_mode", "video")).lower()

# Capture method
CAPTURE_MODE = str(CFG.get("capture_mode", "segment")).lower()
SEG_START_FRAMES = int(CFG.get("seg_start_frames", 3))
SEG_END_FRAMES = int(CFG.get("seg_end_frames", 8))
SEG_MIN_FRAMES = int(CFG.get("seg_min_frames", 8))
SEG_MAX_SECONDS = float(CFG.get("seg_max_seconds", 5.0))
SEG_TRIM_FRAMES = int(CFG.get("seg_trim_frames", 0))
MIN_CONFIDENCE = float(CFG.get("min_confidence", 0.6))

# Gemini fallback
GEMINI_ENABLED = bool(CFG.get("gemini_enabled", True))
GEMINI_API_KEY = CFG.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = str(CFG.get("gemini_model", "gemini-2.5-flash"))
GEMINI_FLUSH_AFTER = int(CFG.get("gemini_flush_after", 2))
GEMINI_PERSIST = bool(CFG.get("gemini_persist", True))
GEMINI_TIMEOUT = float(CFG.get("gemini_timeout", 8.0))

# =========================
# Labels + model
# =========================
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

vlog("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
vlog("Model loaded.")

# Sanity check: the model's output classes MUST match label_map.json (same training run).
try:
    _n_out = int(model.output_shape[-1])
    if _n_out != len(label_map):
        vlog(f"*** WARNING: model outputs {_n_out} classes but label_map.json has "
             f"{len(label_map)} entries. They must come from the SAME training run, or "
             f"predictions will be wrong. Redeploy the matching model.keras + label_map.json. ***")
    _missing = [i for i in range(_n_out) if i not in inv_label_map]
    if _missing:
        vlog(f"*** WARNING: label_map.json is missing class indices {_missing[:15]} "
             f"that the model can output. ***")
except Exception as e:
    vlog("Could not verify model/label_map consistency:", repr(e))

def predict_seq(seq_30x126: np.ndarray) -> Tuple[str, float]:
    x = np.expand_dims(seq_30x126.astype(np.float32), axis=0)
    probs = model.predict(x, verbose=0)[0]
    cls = int(np.argmax(probs))
    # Never crash on an unexpected index (e.g. model/label_map mismatch).
    return inv_label_map.get(cls, f"CLASS_{cls}"), float(np.max(probs))

# =========================
# Gloss DB
# =========================
_db_lock = threading.Lock()

def normalize_gloss_seq(tokens: List[str]) -> str:
    return " ".join([t.strip().upper() for t in tokens if t.strip()])

def load_gloss_db(path: str) -> Dict[str, str]:
    db: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=>" not in line:
                    continue
                left, right = line.split("=>", 1)
                gloss = normalize_gloss_seq(left.strip().split())
                text = right.strip()
                if gloss and text:
                    db[gloss] = text
        vlog(f"Gloss DB loaded: {len(db)} entries")
    except FileNotFoundError:
        vlog(f"Gloss DB not found: {path}")
    return db

gloss_db = load_gloss_db(GLOSS_DB_PATH)

def lookup_longest_suffix(gloss_history: List[str], max_len: int = 12) -> Optional[Tuple[str, str]]:
    n = len(gloss_history)
    L = min(max_len, n)
    for k in range(L, 0, -1):
        seq = normalize_gloss_seq(gloss_history[n - k:n])
        if seq in gloss_db:
            return seq, gloss_db[seq]
    return None

def add_gloss_mapping(gloss_seq: str, english: str, persist: bool = True) -> None:
    gloss_seq = normalize_gloss_seq(gloss_seq.split())
    english = english.strip()
    if not gloss_seq or not english:
        return
    with _db_lock:
        if gloss_db.get(gloss_seq) == english:
            return
        gloss_db[gloss_seq] = english
        if persist:
            try:
                with open(GLOSS_DB_PATH, "a", encoding="utf-8") as f:
                    f.write(f"\n{gloss_seq} => {english}")
            except Exception as e:
                vlog("Gloss DB persist error:", repr(e))

# =========================
# Gemini gloss -> English
# =========================
_gemini_cache: Dict[str, str] = {}
_gemini_cache_lock = threading.Lock()

def gloss_to_english(gloss_tokens: List[str]) -> Optional[str]:
    key_seq = normalize_gloss_seq(gloss_tokens)
    if not key_seq:
        return None
    with _gemini_cache_lock:
        if key_seq in _gemini_cache:
            return _gemini_cache[key_seq]
    if not (GEMINI_ENABLED and GEMINI_API_KEY and requests is not None):
        return None

    prompt = (
        "You are translating a sequence of sign-language GLOSS tokens into one natural, "
        "everyday English sentence. Gloss tokens are uppercase word stems with no grammar. "
        "Return ONLY the English sentence, no quotes, no explanation.\n\n"
        f"GLOSS: {key_seq}\nEnglish:"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        resp = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
            timeout=GEMINI_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip().strip('"').strip()
        if text:
            with _gemini_cache_lock:
                _gemini_cache[key_seq] = text
            if GEMINI_PERSIST:
                add_gloss_mapping(key_seq, text, persist=True)
            vlog(f"Gemini: '{key_seq}' -> '{text}'")
            return text
    except Exception as e:
        vlog("Gemini error:", repr(e))
    return None

# =========================
# Shared state
# =========================
@dataclass
class PerfStats:
    cam_fps: float = 0.0
    mp_fps: float = 0.0
    preview_fps: float = 0.0
    cam_frames: int = 0
    mp_frames: int = 0
    preview_frames: int = 0
    cam_drops: int = 0
    last_mp_ms: float = 0.0

stats = PerfStats()

latest_bgr_lock = threading.Lock()
latest_bgr: Optional[np.ndarray] = None
latest_ts_ms: int = 0

ring_lock = threading.Lock()
# ring stores (ts_ms, kp[126], present[2])
ring: Deque[Tuple[int, np.ndarray, np.ndarray]] = deque(maxlen=RING_MAX)

# =========================
# Camera + MediaPipe worker (IMAGE mode, full-res, handedness-ordered)
# =========================
def _open_capture() -> cv2.VideoCapture:
    backend = CAM_BACKEND
    if backend == "auto":
        if sys.platform.startswith("win"):
            return cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        return cv2.VideoCapture(CAM_INDEX)
    if backend == "dshow":
        return cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(CAM_INDEX, cv2.CAP_MSMF)
    if backend == "v4l2":
        return cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    return cv2.VideoCapture(CAM_INDEX)

def camera_mp_worker(stop_event: threading.Event):
    global latest_bgr, latest_ts_ms

    vlog("Opening camera...")
    cap = _open_capture()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS_REQ)
    if not cap.isOpened():
        vlog("ERROR: camera not opened")
        return

    # In LIVE_STREAM mode, results arrive asynchronously in this callback and we
    # append every one to the ring (no frame dropped on our side).
    def _on_live_result(result, output_image, timestamp_ms):
        kp, present = sf.landmarks_from_result(result)
        with ring_lock:
            ring.append((int(timestamp_ms), kp, present))
        stats.mp_frames += 1

    try:
        if MP_MODE in ("live_stream", "live", "stream"):
            landmarker = sf.make_landmarker(TASK_PATH, mode="live_stream", result_callback=_on_live_result)
        else:
            landmarker = sf.make_landmarker(TASK_PATH, mode=MP_MODE)
        vlog(f"MediaPipe landmarker created ({MP_MODE.upper()} mode).")
    except Exception as e:
        vlog("ERROR creating landmarker:", repr(e))
        cap.release()
        return

    is_live = MP_MODE in ("live_stream", "live", "stream")
    base = time.perf_counter()
    last_ts = 0
    mp_t0 = time.perf_counter()
    mp_cnt = 0

    while not stop_event.is_set():
        ok, bgr = cap.read()
        if not ok or bgr is None:
            stats.cam_drops += 1
            time.sleep(0.005)
            continue

        stats.cam_frames += 1
        ts_ms = int((time.perf_counter() - base) * 1000.0)
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms

        with latest_bgr_lock:
            latest_bgr = bgr
            latest_ts_ms = ts_ms

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # full resolution -> matches training
        if is_live:
            sf.detect_async(landmarker, rgb, ts_ms)   # result handled in callback
        else:
            if MP_MODE == "video":
                result = sf.detect_video(landmarker, rgb, ts_ms)
            else:
                result = sf.detect_image(landmarker, rgb)
            stats.last_mp_ms = (time.perf_counter() - t0) * 1000.0
            kp, present = sf.landmarks_from_result(result)
            with ring_lock:
                ring.append((ts_ms, kp, present))
            stats.mp_frames += 1

        mp_cnt += 1
        now = time.perf_counter()
        if now - mp_t0 >= 1.0:
            stats.mp_fps = mp_cnt / (now - mp_t0)
            mp_t0 = now
            mp_cnt = 0

    vlog("Stopping camera worker...")
    try:
        landmarker.close()
    except Exception:
        pass
    cap.release()
    with latest_bgr_lock:
        latest_bgr = None
        latest_ts_ms = 0
    with ring_lock:
        ring.clear()

# =========================
# Lazy, restartable runtime
# =========================
_runtime_lock = threading.Lock()
_runtime_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_client_count = 0

def ensure_runtime_started():
    global _runtime_thread, _stop_event, _client_count
    with _runtime_lock:
        _client_count += 1
        if _runtime_thread is not None and _runtime_thread.is_alive():
            return
        _stop_event = threading.Event()
        _runtime_thread = threading.Thread(target=camera_mp_worker, args=(_stop_event,), daemon=True)
        _runtime_thread.start()
        vlog(f"Runtime started (clients={_client_count})")

def release_runtime():
    global _client_count, _stop_event, _runtime_thread
    with _runtime_lock:
        _client_count = max(0, _client_count - 1)
        if _client_count == 0 and _stop_event is not None:
            _stop_event.set()
            _runtime_thread = None
            vlog("Runtime stopped (no clients)")

def runtime_is_running() -> bool:
    with _runtime_lock:
        return _runtime_thread is not None and _runtime_thread.is_alive()

# =========================
# Sampling + normalization (identical to training via sign_features)
# =========================
def get_frames_after(ts_ms: int):
    with ring_lock:
        return [(ts, kp, pr) for (ts, kp, pr) in ring if ts > ts_ms]

def _finalize(seq_raw: np.ndarray, pres_raw: np.ndarray) -> np.ndarray:
    if FEATURE_NORMALIZE == "sequence":
        return sf.normalize_sequence(seq_raw, pres_raw)
    return seq_raw.astype(np.float32)

def sample_segment(start_ts_ms: int, end_ts_ms: int, trim_frames: int = 0):
    """Uniformly sample SEQ_LEN frames across the segment, then normalize.
    Returns (seq[SEQ_LEN,126], hand_frames, n_raw)."""
    with ring_lock:
        buf = [(ts, kp, pr) for (ts, kp, pr) in ring if start_ts_ms <= ts <= end_ts_ms]
    if not buf:
        return np.zeros((SEQ_LEN, FEATURES), np.float32), 0, 0
    if trim_frames > 0 and len(buf) > (2 * trim_frames + 2):
        buf = buf[trim_frames: len(buf) - trim_frames]

    n = len(buf)
    idxs = sf.sample_indices(n)
    seq_raw = np.zeros((SEQ_LEN, FEATURES), np.float32)
    pres_raw = np.zeros((SEQ_LEN, sf.HANDS), np.float32)
    hand_frames = 0
    for i, ix in enumerate(idxs):
        seq_raw[i] = buf[ix][1]
        pres_raw[i] = buf[ix][2]
        if buf[ix][2].any():
            hand_frames += 1
    return _finalize(seq_raw, pres_raw), hand_frames, n

def get_window_samples(end_ts_ms: int):
    """Legacy fixed-window sampler (also normalized). Returns (seq, hand_frames)."""
    start_ts_ms = end_ts_ms - TAKE_MS
    with ring_lock:
        buf = [(ts, kp, pr) for (ts, kp, pr) in ring if start_ts_ms <= ts <= end_ts_ms]
    if not buf:
        return np.zeros((SEQ_LEN, FEATURES), np.float32), 0
    w_ts = np.array([x[0] for x in buf], dtype=np.int64)
    targets = np.linspace(start_ts_ms, end_ts_ms, SEQ_LEN).astype(int)
    seq_raw = np.zeros((SEQ_LEN, FEATURES), np.float32)
    pres_raw = np.zeros((SEQ_LEN, sf.HANDS), np.float32)
    hand_frames = 0
    for i, t in enumerate(targets):
        idx = int(np.argmin(np.abs(w_ts - t)))
        seq_raw[i] = buf[idx][1]
        pres_raw[i] = buf[idx][2]
        if buf[idx][2].any():
            hand_frames += 1
    return _finalize(seq_raw, pres_raw), hand_frames
