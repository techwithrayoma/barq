from sqlalchemy import delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.entities.comment import Comment

class CommentRepository:
    """
    Repository for CRUD operations on the Comment table.
    Encapsulates database interactions for Comment entities.
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create_comment(self, comment: Comment) -> Comment:
        """Insert a new comment into the database."""
        self.db_session.add(comment)
        await self.db_session.commit()
        await self.db_session.refresh(comment)
        return comment

    async def get_comments_by_video(
        self, video_id: int, page: int = 1, page_size: int = 20
    ) -> tuple[list[Comment], int]:
        """Return paginated comments and total pages for a given video."""
        
        # Total number of comments
        total_count_query = select(func.count(Comment.id)).where(Comment.video_id == video_id)
        total_count_result = await self.db_session.execute(total_count_query)
        total_count = total_count_result.scalar_one()
        total_pages = (total_count + page_size - 1) // page_size

        # Fetch comments for the current page
        query = (
            select(Comment)
            .where(Comment.video_id == video_id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db_session.execute(query)
        comments = result.scalars().all()

        return comments, total_pages

    async def delete_comments_by_video(self, video_id: int):
        """
        Delete all comments associated with a given video_id.
        """
        stmt = delete(Comment).where(Comment.video_id == video_id)
        await self.db_session.execute(stmt)
        await self.db_session.commit()

    
    async def delete_all_comments(self) -> int:
        """
        Delete all comments from the database.

        Returns:
            int: Number of rows deleted.
        """
        stmt = delete(Comment)
        result = await self.db_session.execute(stmt)
        await self.db_session.commit()
        return result.rowcount