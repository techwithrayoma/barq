import logging
from typing import List

from app.entities.video import Video
from app.entities.comment import Comment
from app.service.video_service import VideoService
from app.service.comment_service import CommentService

class TubeMoodService:
    def __init__(self, db_session, sentiment_service=None):
        self.logger = logging.getLogger(__name__)
        self.video_service = VideoService(db_session)
        self.comment_service = CommentService(db_session, sentiment_service=sentiment_service)

    async def analyze_video(
        self, video_url: str, refresh: bool = False, page: int = 1, page_size: int = 20
    ) -> dict:
        """
        Orchestrates video + comments fetching and returns structured analysis.
        Raises ValueError for expected errors.
        """
        # Video
        video: Video = await self.video_service.fetch_or_create_video(video_url)

        # Comments
        comments: List[Comment] = await self.comment_service.fetch_or_refresh_comments(
            video_id=video.id,
            youtube_id=video.youtube_id,
            refresh=refresh
        )

        # 3Serialize
        serialized_comments = await self.comment_service.serialize_comments(comments)

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_comments = serialized_comments[start_idx:end_idx]
        total_pages = (len(serialized_comments) + page_size - 1) // page_size

        return {
            "video": {
                "video_id": video.id,
                "youtube_id": video.youtube_id,
                "title": video.video_title,
                "published_at": str(video.published_at)
            },
            "comments": paginated_comments,
            "total_pages": total_pages
        }
    
    async def reset_database(self):
        """
        Resets the database by deleting all videos and comments.
        """
        await self.comment_service.reset_comments()
        await self.video_service.reset_videos()
        self.logger.info("Database has been reset.")

    