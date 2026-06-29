"""
Local parity test. Predicts glosses on your recorded videos using the EXACT
same preprocessing the live app uses (app/core/sign_features.py). Run this from
the project root to confirm train/inference parity BEFORE testing the camera.

Usage:
    python tools/test_videos.py /path/to/recordings
        # expects /path/to/recordings/<GLOSS>/<clip>.mp4  -> prints accuracy + confusion
    python tools/test_videos.py /path/to/one_clip.mp4
        # single file -> prints prediction
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core import sign_features as sf

MODEL_PATH = "model.keras"
LABEL_MAP_PATH = "label_map.json"
TASK_PATH = "hand_landmarker.task"
CONFIG_PATH = "config.json"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]

    cfg = json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {}
    normalize = str(cfg.get("feature_normalize", "sequence")).lower()

    label_map = json.load(open(LABEL_MAP_PATH))
    inv = {v: k for k, v in label_map.items()}
    model = tf.keras.models.load_model(MODEL_PATH)

    def predict(path):
        seq, pres = sf.raw_sequence_from_video(path, TASK_PATH)
        if seq is None:
            return None, 0.0
        feat = sf.normalize_sequence(seq, pres) if normalize == "sequence" else seq
        probs = model.predict(np.expand_dims(feat, 0), verbose=0)[0]
        return inv[int(np.argmax(probs))], float(np.max(probs))

    if os.path.isfile(target):
        pred, conf = predict(target)
        print(f"{os.path.basename(target)} -> {pred}  ({conf:.3f})")
        return

    labels = sorted(d for d in os.listdir(target) if os.path.isdir(os.path.join(target, d)))
    total = correct = 0
    confusion = {}
    for label in labels:
        cls_dir = os.path.join(target, label)
        for vid in sorted(os.listdir(cls_dir)):
            if not vid.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                continue
            pred, conf = predict(os.path.join(cls_dir, vid))
            ok = (pred == label)
            total += 1
            correct += int(ok)
            confusion.setdefault(label, {}).setdefault(pred, 0)
            confusion[label][pred] += 1
            flag = "OK " if ok else "XX "
            print(f"{flag}{label:>12} -> {str(pred):>12}  ({conf:.3f})  {vid}")

    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")
        print("\nMisclassifications (true -> predicted):")
        for t in labels:
            for p, c in confusion.get(t, {}).items():
                if p != t:
                    print(f"  {t:>12} -> {str(p):>12} : {c}")


if __name__ == "__main__":
    main()
