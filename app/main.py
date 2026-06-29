from contextlib import asynccontextmanager
from pathlib import Path
from app.core import video_index
from app.core.video_index import init_video_index
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import pages, sign_to_text, text_to_sign, video_lookup
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

STATIC_DIR = BASE_DIR / "static"
ARCHIVE_DIR = BASE_DIR / "archive"
WLASL_JSON_PATH = ARCHIVE_DIR / "WLASL_v0.3.json"
WLASL_VIDEO_DIR = ARCHIVE_DIR / "videos"



@asynccontextmanager
async def lifespan(app: FastAPI):
    # NOTE: the camera + MediaPipe worker is NOT started here anymore.
    # It starts lazily when a client opens the Sign -> Text module (WebSocket
    # connect) and is released when the last client disconnects. See
    # app/core/runtime.py: ensure_runtime_started() / release_runtime().
    
    video_index = init_video_index(
    json_path=str(WLASL_JSON_PATH),
    video_dir=str(WLASL_VIDEO_DIR),
    )

    print(
    "VIDEO LOOKUP:",
    video_index.gloss_count,
    "signs and",
    video_index.total_video_count,
    "videos indexed.",
     )
    
    
    yield


BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount(
    "/archive-videos",
    StaticFiles(directory=str(WLASL_VIDEO_DIR)),
    name="archive_videos",
)
app.include_router(pages.router)
app.include_router(sign_to_text.router)
app.include_router(text_to_sign.router)
app.include_router(video_lookup.router)
