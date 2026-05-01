from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean

from .base import Base

class Comment(Base):
    """
    Represents a comment on a YouTube video in the TubeMood database.

    Attributes:
        id (int): Primary key of the comment in the database.
        video_id (int): Foreign key referencing the video this comment belongs to.
        comment_text (str): Text content of the comment.
        author (str, optional): Author of the comment, if available.
        published_at (datetime, optional): Timestamp when the comment was posted on YouTube.
        sentiment (str, optional): Sentiment of the comment (positive/negative).
        created_at (datetime): Timestamp of when this comment was added to the database.
    """
    
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=False)
    
    # Foreign key to the video this comment belongs to
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    
    # Text of the comment
    comment_text = Column(String, nullable=False)
    
    # Author of the comment, can be null if unknown
    author = Column(String, nullable=True)
    
    # When the comment was published on YouTube, can be null
    published_at = Column(DateTime(timezone=True), nullable=True)
    
    # Sentiment of the comment (e.g., positive, negative, neutral)
    sentiment = Column(String, nullable=True)
    
    # When this comment record was added to the database
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    used_in_training = Column(Boolean, default=False, nullable=False)