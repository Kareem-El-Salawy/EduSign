import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

JSON_PATH = BASE_DIR / "archive" / "WLASL_v0.3.json"
VIDEO_DIR = BASE_DIR / "archive" / "videos"
OUTPUT_PATH = BASE_DIR / "data" / "wlasl_tts_map.json"

# Change this to the signer you selected.
PREFERRED_SIGNER_ID = 10

# True:
# Only the preferred signer is used. Other glosses are missing.
#
# False:
# Another signer is used when the preferred signer is unavailable.
STRICT_ONE_SIGNER = False


def main() -> None:
    local_video_ids = {
        path.stem
        for path in VIDEO_DIR.glob("*.mp4")
    }

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    output = {}

    for entry in dataset:
        gloss = entry.get("gloss", "").strip().lower()

        local_instances = [
            instance
            for instance in entry.get("instances", [])
            if str(instance.get("video_id", "")) in local_video_ids
        ]

        if not local_instances:
            continue

        preferred_instances = [
            instance
            for instance in local_instances
            if instance.get("signer_id") == PREFERRED_SIGNER_ID
        ]

        if preferred_instances:
            selected = preferred_instances[0]
            used_fallback_signer = False

        elif STRICT_ONE_SIGNER:
            continue

        else:
            selected = local_instances[0]
            used_fallback_signer = True

        video_id = str(selected["video_id"])

        output[gloss] = {
            "gloss": gloss,
            "video": f"/archive-videos/{video_id}.mp4",
            "video_id": video_id,
            "signer_id": selected.get("signer_id"),
            "preferred_signer": not used_fallback_signer,
            "fallback_signer": used_fallback_signer
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    preferred_count = sum(
        1 for item in output.values()
        if item["preferred_signer"]
    )

    fallback_count = sum(
        1 for item in output.values()
        if item["fallback_signer"]
    )

    print(f"Saved map: {OUTPUT_PATH}")
    print(f"Total glosses: {len(output)}")
    print(f"Preferred signer videos: {preferred_count}")
    print(f"Fallback signer videos: {fallback_count}")


if __name__ == "__main__":
    main()