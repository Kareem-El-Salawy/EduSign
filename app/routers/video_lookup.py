from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.core.video_index import get_video_index

router = APIRouter(prefix="/api")


@router.get("/videos")
async def search_videos(
    q: str = Query(default="", max_length=200),
    limit: int = Query(default=12, ge=1, le=50),
):
    idx = get_video_index()
    if idx is None:
        return JSONResponse({"error": "index not ready"}, status_code=503)

    results = idx.search(q.strip(), limit=limit)
    return JSONResponse({
        "q": q,
        "count": len(results),
        "results": results,
        "stats": {
            "gloss_count": idx.gloss_count,
            "video_count": idx.total_video_count,
            "nlp": idx.nlp_info,
        },
    })
