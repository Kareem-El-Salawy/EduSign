import asyncio
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse

from app.core import runtime as rt

router = APIRouter()


@router.get("/health")
def health():
    with rt.ring_lock:
        ring_len = len(rt.ring)
    with rt.latest_bgr_lock:
        ts = rt.latest_ts_ms
    return JSONResponse({
        "running": rt.runtime_is_running(),
        "cam_fps": rt.stats.cam_fps,
        "mp_fps": rt.stats.mp_fps,
        "ring_len": ring_len,
        "latest_ts_ms": ts,
        "capture_mode": rt.CAPTURE_MODE,
        "gloss_db_entries": len(rt.gloss_db),
        "gemini_enabled": bool(rt.GEMINI_ENABLED and rt.GEMINI_API_KEY),
    })


@router.get("/preview.mjpg")
def preview_mjpg():
    boundary = "frame"

    def gen():
        prev_t = 0.0
        while True:
            now = time.perf_counter()
            if now - prev_t < (1.0 / rt.PREVIEW_FPS):
                time.sleep(0.001)
                continue
            prev_t = now

            with rt.latest_bgr_lock:
                frame = None if rt.latest_bgr is None else rt.latest_bgr.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.resize(frame, (rt.PREVIEW_W, rt.PREVIEW_H), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            payload = jpg.tobytes()
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" +
                   payload + b"\r\n")
            rt.stats.preview_frames += 1

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # Opening the Sign -> Text module starts the camera/MP runtime (ref-counted).
    rt.ensure_runtime_started()

    recording = False
    gloss_history: List[str] = []
    sentence_words: List[str] = []
    pending_start = 0  # index of first not-yet-translated gloss
    take_id = 0
    gemini_inflight = False
    driver_task: Optional[asyncio.Task] = None
    consumer_task: Optional[asyncio.Task] = None
    # Finished segments waiting to be classified. The segmenter PUTS here and keeps
    # watching your hands; a separate consumer classifies in the background, so
    # signing is never blocked by recognition.
    seg_queue: "asyncio.Queue" = asyncio.Queue()

    async def safe_send(obj: Dict):
        try:
            await ws.send_json(obj)
        except Exception as e:
            rt.vlog("WS send failed:", repr(e))

    async def stop_driver():
        nonlocal driver_task, consumer_task
        for t in (driver_task, consumer_task):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        driver_task = None
        consumer_task = None

    async def flush_pending(reason: str = "flush"):
        """Translate the run of glosses since the last match (awaited; used on stop)."""
        nonlocal pending_start
        pending = gloss_history[pending_start:]
        if not pending:
            return
        english = await asyncio.to_thread(rt.gloss_to_english, pending)
        if english:
            sentence_words.append(english)
            pending_start = len(gloss_history)
            await safe_send({
                "type": "spoken", "gloss_seq": rt.normalize_gloss_seq(pending),
                "text": english, "source": "gemini",
                "gloss": gloss_history, "spoken": sentence_words,
            })

    def trigger_gemini():
        """Translate pending glosses via Gemini in the BACKGROUND so the classifier
        consumer is never blocked waiting on the network."""
        nonlocal gemini_inflight, pending_start
        pending = gloss_history[pending_start:]
        if not pending or gemini_inflight:
            return
        flushed_to = len(gloss_history)
        gemini_inflight = True

        async def _do():
            nonlocal gemini_inflight, pending_start
            try:
                english = await asyncio.to_thread(rt.gloss_to_english, pending)
                if english:
                    sentence_words.append(english)
                    pending_start = max(pending_start, flushed_to)
                    await safe_send({
                        "type": "spoken", "gloss_seq": rt.normalize_gloss_seq(pending),
                        "text": english, "source": "gemini",
                        "gloss": gloss_history, "spoken": sentence_words,
                    })
            finally:
                gemini_inflight = False

        asyncio.create_task(_do())

    async def emit_gloss(gloss_word: str, conf: float):
        """Append a recognized gloss; the gloss is sent immediately, sentence
        translation (local DB, then background Gemini) follows without blocking."""
        nonlocal pending_start, take_id
        gloss_history.append(gloss_word)
        take_id += 1
        await safe_send({
            "type": "word", "word": gloss_word, "conf": conf,
            "gloss": gloss_history, "spoken": sentence_words, "take_id": take_id,
        })

        hit = rt.lookup_longest_suffix(gloss_history, max_len=12)
        if hit is not None:
            gloss_seq, english = hit
            sentence_words.append(english)
            pending_start = len(gloss_history)
            await safe_send({
                "type": "spoken", "gloss_seq": gloss_seq, "text": english,
                "source": "local", "gloss": gloss_history, "spoken": sentence_words,
            })
        elif len(gloss_history) - pending_start >= rt.GEMINI_FLUSH_AFTER:
            trigger_gemini()  # background, non-blocking

    async def classifier_consumer():
        """Background worker: classify queued segments one at a time (serialized so
        the model is used by a single thread) while the segmenter keeps capturing.
        A failure on one segment must never kill the worker."""
        rt.vlog("classifier_consumer started")
        while True:
            seq, hand_frames, n_raw = await seg_queue.get()
            try:
                if n_raw < rt.SEG_MIN_FRAMES or hand_frames == 0:
                    rt.vlog(f"segment dropped: {n_raw} frames, {hand_frames} with hands")
                    await safe_send({"type": "dropped", "reason": "too_short", "frames": n_raw})
                else:
                    gloss_word, conf = await asyncio.to_thread(rt.predict_seq, seq)
                    rt.vlog(f"segment: {n_raw} frames -> {gloss_word} (conf {conf:.2f})")
                    if conf < rt.MIN_CONFIDENCE:
                        await safe_send({"type": "low_conf", "word": gloss_word, "conf": conf})
                    else:
                        await emit_gloss(gloss_word, conf)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                rt.vlog("classifier error (skipped):", repr(e))
            finally:
                seg_queue.task_done()
                await safe_send({"type": "queue", "depth": seg_queue.qsize()})

    async def enqueue_segment(start_ts: int, end_ts: int):
        """Sample the finished segment (fast) and hand it to the background classifier."""
        seq, hand_frames, n_raw = rt.sample_segment(start_ts, end_ts, rt.SEG_TRIM_FRAMES)
        await seg_queue.put((seq, hand_frames, n_raw))
        await safe_send({"type": "queue", "depth": seg_queue.qsize()})


    async def segment_driver():
        """Presence-gated capture: start on hands-in, classify on hands-out."""
        nonlocal recording
        rt.vlog("segment_driver started")

        present_run = 0
        absent_run = 0
        run_start_ts: Optional[int] = None  # start of the current present-run
        onset_ts: Optional[int] = None      # locked segment start (survives flicker)
        offset_ts: Optional[int] = None
        state = "LISTENING"

        with rt.latest_bgr_lock:
            last_ts = rt.latest_ts_ms  # ignore stale buffer before we started

        await safe_send({"type": "phase", "phase": "LISTENING", "take_id": take_id})

        while recording:
            frames = rt.get_frames_after(last_ts)
            if not frames:
                await asyncio.sleep(0.02)
                continue

            for (ts, kp, present) in frames:
                last_ts = ts
                has = bool(np.asarray(present).any())

                if has:
                    if present_run == 0:
                        run_start_ts = ts
                    present_run += 1
                    absent_run = 0
                else:
                    if absent_run == 0:
                        offset_ts = ts
                    absent_run += 1
                    present_run = 0

                if state == "LISTENING":
                    if present_run >= rt.SEG_START_FRAMES:
                        state = "CAPTURING"
                        onset_ts = run_start_ts   # lock start; do NOT move it on flicker
                        offset_ts = None
                        await safe_send({"type": "phase", "phase": "CAPTURING", "take_id": take_id})

                elif state == "CAPTURING":
                    too_long = onset_ts is not None and (ts - onset_ts) > rt.SEG_MAX_SECONDS * 1000
                    ended = absent_run >= rt.SEG_END_FRAMES
                    if too_long or ended:
                        seg_end = offset_ts if (ended and offset_ts is not None) else ts
                        if onset_ts is not None:
                            await enqueue_segment(onset_ts, seg_end)  # hand off, keep going
                        state = "LISTENING"
                        present_run = absent_run = 0
                        onset_ts = offset_ts = run_start_ts = None
                        await safe_send({"type": "phase", "phase": "LISTENING", "take_id": take_id})

            await asyncio.sleep(0.01)

        rt.vlog("segment_driver stopped")

    async def fixed_driver():
        """Legacy fixed-2s trailing window capture."""
        nonlocal recording
        take_start_ts: Optional[int] = None
        await safe_send({"type": "phase", "phase": "CAPTURING", "take_id": take_id})

        while recording:
            with rt.latest_bgr_lock:
                now_ts = rt.latest_ts_ms
            if now_ts <= 0:
                await asyncio.sleep(0.05)
                continue
            if take_start_ts is None:
                take_start_ts = now_ts
            if (now_ts - take_start_ts) < rt.TAKE_MS:
                await asyncio.sleep(0.02)
                continue

            await safe_send({"type": "phase", "phase": "PROCESSING", "take_id": take_id})
            seq, hand_frames = await asyncio.to_thread(rt.get_window_samples, now_ts)
            if hand_frames > 0:
                gloss_word, conf = await asyncio.to_thread(rt.predict_seq, seq)
                if conf >= rt.MIN_CONFIDENCE:
                    await emit_gloss(gloss_word, conf)
            take_start_ts = now_ts
            await safe_send({"type": "phase", "phase": "CAPTURING", "take_id": take_id})
            await asyncio.sleep(0.02)

    await safe_send({"type": "status", "recording": False, "gloss": [], "spoken": [], "phase": "IDLE"})

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") != "control":
                continue
            cmd = msg.get("cmd")

            if cmd == "start":
                recording = True
                gloss_history = []
                sentence_words = []
                pending_start = 0
                take_id = 0
                gemini_inflight = False
                # fresh queue for this session
                while not seg_queue.empty():
                    try:
                        seg_queue.get_nowait(); seg_queue.task_done()
                    except Exception:
                        break
                await safe_send({"type": "status", "recording": True, "gloss": [], "spoken": [], "phase": "LISTENING"})
                await stop_driver()
                consumer_task = asyncio.create_task(classifier_consumer())
                if rt.CAPTURE_MODE == "fixed":
                    driver_task = asyncio.create_task(fixed_driver())
                else:
                    driver_task = asyncio.create_task(segment_driver())

            elif cmd == "stop":
                recording = False
                # let the segmenter stop, then finish classifying whatever is queued
                if driver_task and not driver_task.done():
                    driver_task.cancel()
                    try:
                        await driver_task
                    except asyncio.CancelledError:
                        pass
                    driver_task = None
                try:
                    await asyncio.wait_for(seg_queue.join(), timeout=10.0)
                except asyncio.TimeoutError:
                    rt.vlog("stop: classifier queue drain timed out")
                await flush_pending("manual_stop")
                await stop_driver()
                await safe_send({"type": "status", "recording": False, "gloss": gloss_history, "spoken": sentence_words, "phase": "IDLE"})

            elif cmd == "edit_gloss":
                # Manual correction: replace one recognized gloss.
                i = msg.get("index")
                val = (msg.get("value") or "").strip()
                if isinstance(i, int) and 0 <= i < len(gloss_history) and val:
                    gloss_history[i] = val.upper().replace(" ", "_")
                    rt.vlog(f"gloss[{i}] corrected -> {gloss_history[i]}")
                    await safe_send({"type": "gloss_update", "gloss": gloss_history, "spoken": sentence_words})

            elif cmd == "delete_gloss":
                i = msg.get("index")
                if isinstance(i, int) and 0 <= i < len(gloss_history):
                    removed = gloss_history.pop(i)
                    if i < pending_start:
                        pending_start -= 1
                    rt.vlog(f"gloss[{i}] deleted ({removed})")
                    await safe_send({"type": "gloss_update", "gloss": gloss_history, "spoken": sentence_words})

            elif cmd == "clear":
                gloss_history.clear()
                sentence_words.clear()
                pending_start = 0
                await safe_send({"type": "gloss_update", "gloss": gloss_history, "spoken": sentence_words})

    except WebSocketDisconnect:
        rt.vlog("WS disconnected")
    finally:
        recording = False
        await stop_driver()
        rt.release_runtime()
