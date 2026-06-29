from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"


def read_html(filename: str) -> HTMLResponse:
    path = STATIC_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        return HTMLResponse(file.read())


@router.get("/")
def home():
    return read_html("index.html")


@router.get("/sign-to-text")
def sign_to_text_page():
    return read_html("sign_to_text.html")


@router.get("/text-to-sign")
def text_to_sign_page():
    return read_html("text_to_sign.html")


@router.get("/video-lookup")
def video_lookup_page():
    return read_html("video_lookup.html")
