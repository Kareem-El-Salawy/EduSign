import json
import os
import re
from pathlib import Path
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GLOSS_DB_PATH = PROJECT_ROOT / "data" / "gloss_db.json"
WLASL_TTS_MAP_PATH = PROJECT_ROOT / "data" / "wlasl_tts_map.json"
WLASL_VIDEO_DIR = PROJECT_ROOT / "archive" / "videos"

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


router = APIRouter(prefix="/api")

BASE_DIR = Path(__file__).resolve().parents[2]
GLOSS_DB_PATH = BASE_DIR / "data" / "gloss_db.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

IGNORED_WORDS = {
    "can", "could", "would", "should",
    "the", "a", "an",
    "is", "are", "am", "was", "were",
    "to", "for", "of", "and",
    "like", "want"
}


def load_gloss_db():
    with open(GLOSS_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_wlasl_tts_map() -> dict:
    if not WLASL_TTS_MAP_PATH.is_file():
        print("WLASL Text-to-Sign map not found:", WLASL_TTS_MAP_PATH)
        return {}

    with open(WLASL_TTS_MAP_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
    
WLASL_TTS_MAP = load_wlasl_tts_map()    
    
def build_combined_tts_db(local_gloss_db: dict, wlasl_map: dict):
    """
    Builds the Text-to-Sign vocabulary from WLASL while keeping
    aliases from the original gloss_db.json.
    """

    combined = {}

    # Add every available WLASL gloss.
    for key, data in wlasl_map.items():
        gloss = str(data.get("gloss") or key).strip()
        video = data.get("video")

        if not gloss or not video:
            continue

        combined[gloss.lower()] = {
            "gloss": gloss,
            "video": video,
            "video_id": data.get("video_id"),
            "signer_id": data.get("signer_id"),
            "preferred_signer": data.get(
                "preferred_signer",
                False
            ),
            "fallback_signer": data.get(
                "fallback_signer",
                False
            )
        }

    # Keep aliases from the original local database.
    for alias, local_data in local_gloss_db.items():
        official_gloss = str(
            local_data.get("gloss", "")
        ).strip()

        official_key = official_gloss.lower()

        if official_key in combined:
            alias_data = dict(combined[official_key])
            combined[alias.lower()] = alias_data

        # Keep the old local video only when no WLASL video exists.
        elif local_data.get("video"):
            combined[alias.lower()] = {
                "gloss": official_gloss,
                "video": local_data["video"],
                "video_id": None,
                "signer_id": None,
                "preferred_signer": False,
                "fallback_signer": False
            }

    return combined




def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_available_gloss_map(gloss_db: dict):
    available = {}

    for data in gloss_db.values():
        gloss = str(data.get("gloss", "")).strip()

        if not gloss:
            continue

        gloss_key = gloss.lower()

        # Store the complete entry, including signer information.
        available[gloss_key] = dict(data)

    return available

def create_match(input_text: str, data: dict):
    return {
        "input_text": input_text,
        "gloss": data["gloss"],
        "video": data["video"],
        "video_id": data.get("video_id"),
        "signer_id": data.get("signer_id"),
        "preferred_signer": data.get(
            "preferred_signer",
            False
        ),
        "fallback_signer": data.get(
            "fallback_signer",
            False
        )
    }

def separate_missing_videos(matched):
    playable_matches = []
    missing_video_words = []

    for item in matched:
        video_url = item.get("video", "")

        if video_file_exists(video_url):
            playable_matches.append(item)
        else:
            missing_video_words.append(
                item.get("input_text")
                or item.get("gloss")
                or video_url
            )

    return playable_matches, missing_video_words

def match_text_to_glosses(words, gloss_db, max_phrase_len=3):
    results = []
    missing_words = []

    i = 0

    while i < len(words):
        matched_phrase = None
        matched_data = None

        max_len = min(max_phrase_len, len(words) - i)

        for length in range(max_len, 0, -1):
            phrase = " ".join(words[i:i + length])

            if phrase in gloss_db:
                matched_phrase = phrase
                matched_data = gloss_db[phrase]
                break

        if matched_phrase:
            results.append(
                create_match(
                    input_text=matched_phrase,
                    data=matched_data
                )
            )
            i += len(matched_phrase.split())
        else:
            if words[i] not in IGNORED_WORDS:
                missing_words.append(words[i])
            i += 1

    return results, missing_words

def attach_wlasl_videos(matched: list[dict]):
    playable = []
    missing_videos = []

    for item in matched:
        gloss = str(item.get("gloss", "")).strip().lower()
        wlasl_item = WLASL_TTS_MAP.get(gloss)

        if not wlasl_item:
            missing_videos.append(
                item.get("input_text") or item.get("gloss")
            )
            continue

        updated_item = dict(item)

        updated_item["video"] = wlasl_item["video"]
        updated_item["video_id"] = wlasl_item["video_id"]
        updated_item["signer_id"] = wlasl_item["signer_id"]
        updated_item["fallback_signer"] = wlasl_item["fallback_signer"]

        playable.append(updated_item)

    return playable, missing_videos

def video_file_exists(video_url: str) -> bool:
    if not video_url:
        return False

    if video_url.startswith("/static/"):
        relative_path = video_url.replace("/static/", "", 1)
        local_path = PROJECT_ROOT / "static" / relative_path

    elif video_url.startswith("/archive-videos/"):
        relative_path = video_url.replace("/archive-videos/", "", 1)
        local_path = WLASL_VIDEO_DIR / relative_path

    elif video_url.startswith(("http://", "https://")):
        return True

    else:
        local_path = PROJECT_ROOT / video_url.lstrip("/")

    exists = local_path.is_file()

    print(
        "TEXT-TO-SIGN VIDEO CHECK:",
        video_url,
        "->",
        local_path,
        "exists:",
        exists
    )

    return exists


def convert_sentence_with_gemini(text: str, gloss_db: dict):
    """
    Uses Gemini to convert natural English into available glosses only.
    If Gemini fails or API key is missing, returns None.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
     print("GEMINI SKIPPED: GEMINI_API_KEY is missing.")
     return None

    if genai is None:
     print("GEMINI SKIPPED: google-genai failed to import.")
     return None

    print("GEMINI: API key found.")
    print("GEMINI: Using model:", GEMINI_MODEL)

    available_gloss_map = build_available_gloss_map(gloss_db)
    available_glosses = [data["gloss"] for data in available_gloss_map.values()]

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are converting English text into a simple sign-language gloss sequence for a student demo.

Rules:
1. Use ONLY the available glosses listed below.
2. Do not invent new glosses.
3. Remove unnecessary English helper words such as can, could, would, the, a, an, is, are, to.
4. Prefer meaning over exact word matching.
5. Return JSON only.
6. The glosses must match the available gloss names exactly.

Available glosses:
{json.dumps(available_glosses, ensure_ascii=False)}
print(
    "TEXT-TO-SIGN: Sending",
    len(available_glosses),
    "available WLASL glosses to Gemini."
)

Input sentence:
{text}

Return exactly this JSON format:
{{
  "glosses": [],
  "ignored_words": [],
  "missing_meaning": []
}}
"""

    try:
        print(f"TEXT-TO-SIGN: Attempting Gemini request using model {GEMINI_MODEL}.")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response")

        print("TEXT-TO-SIGN: Gemini response received.")
        data = json.loads(response.text)

        matched = []

        for gloss in data.get("glosses", []):
            gloss_key = gloss.lower()

            if gloss_key in available_gloss_map:
                matched.append(
                    create_match(
                        input_text=gloss,
                        data=available_gloss_map[gloss_key]
                    )
                )

        return {
            "matched": matched,
            "ignored_words": data.get("ignored_words", []),
            "missing_meaning": data.get("missing_meaning", [])
        }

    except Exception as error:
        print("TEXT-TO-SIGN GEMINI ERROR:")
        print(type(error).__name__)
        print(str(error))
        return None


@router.post("/text-to-sign")
async def text_to_sign(payload: dict):
    text = (payload.get("text") or "").strip()

    if not text:
        return JSONResponse({
            "text": text,
            "method": "empty",
            "glosses": [],
            "videos": [],
            "matched": [],
            "missing_words": [],
            "ignored_words": [],
            "missing_meaning": []
        })

    local_gloss_db = load_gloss_db()
    wlasl_map = load_wlasl_tts_map()

    gloss_db = build_combined_tts_db(
        local_gloss_db=local_gloss_db,
        wlasl_map=wlasl_map
    )

    print(
        "TEXT-TO-SIGN:",
        len(gloss_db),
        "input mappings are available."
    )
    words = clean_text(text)

    # Use local matching for a single known word.
    direct_matches, direct_missing_words = match_text_to_glosses(
        words,
        gloss_db
    )

    direct_playable, direct_missing_videos = separate_missing_videos(
        direct_matches
    )

    # Use direct matching when every meaningful word was matched
    # and every required video exists.
    if (
        direct_playable
        and not direct_missing_words
        and not direct_missing_videos
    ):
        return JSONResponse({
            "text": text,
            "method": "direct_database_match",
            "glosses": [
                item["gloss"]
                for item in direct_playable
            ],
            "videos": [
                item["video"]
                for item in direct_playable
            ],
            "matched": direct_playable,
            "missing_words": [],
            "ignored_words": [],
            "missing_meaning": []
        })

    # Gemini is only called when interpretation is needed.

    

    # 2. Try Gemini
    gemini_result = convert_sentence_with_gemini(text, gloss_db)

    if gemini_result and len(gemini_result["matched"]) > 0:
        all_matched = gemini_result["matched"]

        matched, missing_video_words = separate_missing_videos(all_matched)

        missing_words = list(dict.fromkeys(
            gemini_result.get("missing_meaning", [])
            + missing_video_words
        ))

        return JSONResponse({
            "text": text,
            "method": "gemini",
            "glosses": [item["gloss"] for item in matched], 
            "videos": [item["video"] for item in matched],
            "matched": matched,
            "missing_words": missing_words,
            "ignored_words": gemini_result.get("ignored_words", []),
            "missing_meaning": gemini_result.get("missing_meaning", [])
        })

    # 3. Fallback to old rule-based system
    print("TEXT-TO-SIGN: Using rule-based fallback.")
    words = clean_text(text)
    matched, missing_words = match_text_to_glosses(words, gloss_db)

    return JSONResponse({
        "text": text,
        "method": "rule_based_fallback",
        "glosses": [item["gloss"] for item in matched],
        "videos": [item["video"] for item in matched],
        "matched": matched,
        "missing_words": missing_words,
        "ignored_words": [],
        "missing_meaning": missing_words
    })