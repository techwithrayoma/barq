from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime

from .base import Base

class Video(Base):
    """
    Represents a YouTube video in the TubeMood database.

    Attributes:
        id (int): Primary key of the video in the database.
        youtube_id (str): Unique identifier for the video from YouTube.
        video_title (str): Title of the video.
        published_at (datetime, optional): Timestamp when the video was published on YouTube.
        created_at (datetime): Timestamp of when this video was added to the database.
    """
    
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=False)
    
    # Unique YouTube ID for the video
    youtube_id = Column(String, unique=True, index=False, nullable=False)
    
    # Title of the video
    video_title = Column(String, nullable=False)
    
    # When the video was published on YouTube, can be null if unknown
    published_at = Column(DateTime(timezone=True), nullable=True)
    
    # When this video record was added to the database
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
