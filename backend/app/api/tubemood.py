from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.db.database import DbSession
from app.service.tubemood_service import TubeMoodService
from .schemas.tubemood import SentimentAnalysisRequest
from app.core.enums import ResponseSignal

router = APIRouter(
    prefix="/tubemood",
    tags=["TubeMood"]
)

@router.post("/sentiment-analysis")
async def sentiment_analysis(
    request: Request,
    payload: SentimentAnalysisRequest,
    db: DbSession
):
    """
    Fetch YouTube comments and analyze a video.
    `refresh=True` fetches new comments from YouTube and updates the DB.
    """
    sentiment_service = request.app.sentiment_service

    service = TubeMoodService(db_session=db, sentiment_service=sentiment_service)

    try:
        result = await service.analyze_video(
            video_url=payload.video_url,
            refresh=payload.refresh
        )
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        # Handles any unhandled exceptions, returning structured response
        return JSONResponse(
            status_code=500,
            content={
                "signal": ResponseSignal.VIDEO_ANALYSIS_FAILED.value,
                "message": f"Unexpected error: {str(e)}"
            }
        )


@router.post("/reset_database")
async def reset_database(
    request: Request,
    db: DbSession
):
    """
    Fetch YouTube comments and analyze a video.
    `refresh=True` fetches new comments from YouTube and updates the DB.
    """
    sentiment_service = request.app.sentiment_service

    service = TubeMoodService(db_session=db, sentiment_service=sentiment_service)

    try:
        result = await service.reset_database(
        )
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        # Handles any unhandled exceptions, returning structured response
        return JSONResponse(
            status_code=500,
            content={
                "signal": ResponseSignal.VIDEO_ANALYSIS_FAILED.value,
                "message": f"Unexpected error: {str(e)}"
            }
        )