from pydantic import BaseModel
from typing import Optional

class SentimentAnalysisRequest(BaseModel):
    video_url: str
    refresh: Optional[bool] = False
