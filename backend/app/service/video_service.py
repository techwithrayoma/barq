import logging

from app.entities.video import Video
from app.repository.video_repository import VideoRepository
from app.service.youtube_service import YouTubeService
from app.core.enums import ErrorMessages

class VideoService:
    def __init__(self, db_session):
        self.logger = logging.getLogger(__name__)
        self.youtube_service = YouTubeService()
        self.video_repo = VideoRepository(db_session=db_session)

    async def fetch_or_create_video(self, video_url: str) -> Video:
        """
        Fetch video from DB or create it from YouTube metadata.
        Raises ValueError for invalid URLs or missing data.
        """
        self._validate_url(video_url)
        video_id = self.youtube_service.get_video_id(video_url)
        if not video_id:
            raise ValueError(ErrorMessages.INVALID_YOUTUBE_URL)

        video = await self.video_repo.get_video_by_youtube_id(video_id)
        if video:
            self.logger.info(f"Video {video_id} already exists in DB")
            return video

        metadata = await self.youtube_service.fetch_video_info(video_id)
        if not metadata:
            raise ValueError(ErrorMessages.VIDEO_NOT_FOUND)

        video = Video(
            youtube_id=metadata["video_id"],
            video_title=metadata["title"],
            published_at=metadata.get("published_at")
        )
        video = await self.video_repo.get_or_create_video(video)
        self.logger.info(f"Video {video_id} created in DB")
        return video

    def _validate_url(self, video_url: str):
        if not video_url.startswith(("http://", "https://")):
            raise ValueError(ErrorMessages.INVALID_URL_FORMAT)
        if "youtube.com/watch" not in video_url and "youtu.be/" not in video_url:
            raise ValueError(ErrorMessages.INVALID_YOUTUBE_URL)

    async def reset_videos(self):
        """
        Deletes all videos and associated comments from the database.
        """
        deleted_count = await self.video_repo.delete_all_videos()
        self.logger.info(f"Deleted {deleted_count} videos from the database.")