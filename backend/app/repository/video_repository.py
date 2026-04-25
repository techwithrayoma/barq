from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from app.entities.video import Video

class VideoRepository:
    """
    Repository for CRUD operations on Video table.
    Encapsulates database interactions for Video entities.
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create_video(self, video: Video) -> Video:
        """Insert a new video into the database."""
        self.db_session.add(video)
        await self.db_session.commit()
        await self.db_session.refresh(video)
        return video

    async def get_video_by_youtube_id(self, youtube_id: str) -> Video | None:
        """Fetch a video by its YouTube ID."""
        query = select(Video).where(Video.youtube_id == youtube_id)
        result = await self.db_session.execute(query)
        return result.scalar_one_or_none()

    async def get_or_create_video(self, video: Video) -> Video:
        """
        Get video if it exists, else create a new one.
        """
        existing = await self.get_video_by_youtube_id(video.youtube_id)
        if existing:
            return existing
        return await self.create_video(video)

    async def delete_all_videos(self) -> int:
        """Delete all videos from the database.

        Returns:
            int: Number of rows deleted.
        """
        query = delete(Video)
        result = await self.db_session.execute(query)
        await self.db_session.commit()
        return result.rowcount