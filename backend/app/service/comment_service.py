import logging
from typing import List

from app.entities.comment import Comment
from app.repository.comment_repository import CommentRepository
from app.service.youtube_service import YouTubeService
from app.core.enums import ErrorMessages

class CommentService:
    def __init__(self, db_session, sentiment_service=None):
        self.logger = logging.getLogger(__name__)
        self.youtube_service = YouTubeService()
        self.comment_repo = CommentRepository(db_session=db_session)
        self.sentiment_service = sentiment_service
        
    async def fetch_or_refresh_comments(self, video_id: int, youtube_id: str, refresh: bool) -> List[Comment]:
        """
        Fetch comments for a video. Refresh from YouTube if requested.
        Raises ValueError if video/comments cannot be retrieved.
        """
        self.logger.debug(f"Fetching comments for video {youtube_id}, refresh={refresh}")

        existing_comments, _ = await self.comment_repo.get_comments_by_video(video_id, page=1, page_size=1)

        if existing_comments and not refresh:
            self.logger.info(f"Returning cached comments for video {youtube_id}")
            comments, _ = await self.comment_repo.get_comments_by_video(video_id, page=1, page_size=1000)
            return comments

        # Refresh or first-time fetch
        if refresh or not existing_comments:
            self.logger.info(f"Refreshing comments for video {youtube_id}")
            await self.comment_repo.delete_comments_by_video(video_id)

            comments_data = await self.youtube_service.fetch_all_comments(youtube_id)
            if comments_data is None:
                raise ValueError(ErrorMessages.VIDEO_NOT_FOUND)

            comments: List[Comment] = []
            for c in comments_data:
                comment_text = c.get("text")
                sentiment = self.sentiment_service.predict(comment_text)

                comment = Comment(
                    video_id=video_id,
                    comment_text=c.get("text"),
                    published_at=c.get("published_at"),
                    author=c.get("author"),
                    sentiment=sentiment
                )
                await self.comment_repo.create_comment(comment)
                comments.append(comment)

            self.logger.info(f"Inserted {len(comments)} comments into DB")
            return comments

    async def serialize_comments(self, comments: List[Comment]) -> List[dict]:
        return [
            {
                "comment_id": com.id,
                "video_id": com.video_id,
                "text": com.comment_text,
                "author": com.author,
                "sentiment": com.sentiment,
                "created_at": com.created_at.isoformat() if com.created_at else None
            }
            for com in comments
        ]

    async def reset_comments(self):
        """
        Deletes all comments from the database.
        """
        deleted_count = await self.comment_repo.delete_all_comments()
        self.logger.info(f"Deleted {deleted_count} comments from the database.")
    
