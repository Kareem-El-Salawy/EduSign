import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

JSON_PATH = BASE_DIR / "archive" / "WLASL_v0.3.json"
VIDEO_DIR = BASE_DIR / "archive" / "videos"


def main() -> None:
    if not JSON_PATH.is_file():
        raise FileNotFoundError(f"WLASL JSON not found: {JSON_PATH}")

    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(f"WLASL video folder not found: {VIDEO_DIR}")

    local_video_ids = {
        path.stem
        for path in VIDEO_DIR.glob("*.mp4")
    }

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    signer_glosses = defaultdict(set)
    signer_video_counts = Counter()

    for entry in dataset:
        gloss = entry.get("gloss", "").strip().lower()

        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id", ""))
            signer_id = instance.get("signer_id")

            if not gloss or signer_id is None:
                continue

            if video_id not in local_video_ids:
                continue

            signer_glosses[signer_id].add(gloss)
            signer_video_counts[signer_id] += 1

    ranked_signers = sorted(
        signer_glosses,
        key=lambda signer: len(signer_glosses[signer]),
        reverse=True
    )

    print("\nTop WLASL signers available locally:\n")

    for signer_id in ranked_signers[:20]:
        print(
            f"Signer {signer_id}: "
            f"{len(signer_glosses[signer_id])} unique glosses, "
            f"{signer_video_counts[signer_id]} videos"
        )


if __name__ == "__main__":
    main()