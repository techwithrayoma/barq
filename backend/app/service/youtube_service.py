from datetime import datetime, timezone
import aiohttp

from app.core.config import get_settings

# Load configuration from environment or .env
settings = get_settings()
YOUTUBE_API_KEY = settings.YOUTUBE_API_KEY

class YouTubeService:
    def __init__(self):
        self.api_key = YOUTUBE_API_KEY  

    @staticmethod
    def get_video_id(url: str):
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url)

        # Case 1: youtu.be short link
        if "youtu.be" in parsed.netloc:
            return parsed.path.strip("/")

        # Case 2: YouTube Shorts link
        if "/shorts/" in parsed.path:
            return parsed.path.split("/shorts/")[1].split("?")[0]

        # Case 3: Normal YouTube link with ?v=
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

        return None


    async def fetch_video_info(self, video_id: str):
        """Fetch video metadata from YouTube API"""
        api_url = (
            f"https://www.googleapis.com/youtube/v3/videos"
            f"?part=snippet,statistics&id={video_id}&key={self.api_key}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if not data.get("items"):
                    return None

                item = data["items"][0]
                snippet = item["snippet"]
                stats = item.get("statistics", {})

                published_at_str = snippet.get("publishedAt")
                published_at = None
                if published_at_str:
                    published_at = datetime.strptime(
                        published_at_str, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)

                return {
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "author": snippet.get("channelTitle"),
                    "views": int(stats.get("viewCount", 0)),
                    "published_at": published_at,
                }

    async def fetch_all_comments(self, video_id: str):
        """Fetch all top-level comments for a YouTube video using pagination"""
        comments = []
        next_page_token = None

        async with aiohttp.ClientSession() as session:
            while True:
                api_url = (
                    f"https://www.googleapis.com/youtube/v3/commentThreads"
                    f"?part=snippet"
                    f"&videoId={video_id}"
                    f"&maxResults=100"
                    f"&key={self.api_key}"
                )
                if next_page_token:
                    api_url += f"&pageToken={next_page_token}"

                async with session.get(api_url) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    for item in data.get("items", []):
                        top_comment = item["snippet"]["topLevelComment"]["snippet"]
                        comments.append({
                            "comment_id": item["snippet"]["topLevelComment"]["id"],
                            "author": top_comment["authorDisplayName"],
                            "text": top_comment["textDisplay"],
                            "like_count": top_comment.get("likeCount", 0),
                            "published_at": datetime.strptime(
                                top_comment["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                            ).replace(tzinfo=timezone.utc),
                        })

                    next_page_token = data.get("nextPageToken")
                    if not next_page_token:
                        break

        return comments