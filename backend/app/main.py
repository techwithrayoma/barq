import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from app.api.tubemood import router as tubemood_router
from app.api.analytics import router as analytics_router
from app.core.logging_config import configure_logging, logLevels
from app.service.sentiment_service import SentimentService

# Configure logging
configure_logging(logLevels.DEBUG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.sentiment_service = SentimentService()
    logger.info("Sentiment model loaded")
    yield
    # Shutdown (optional cleanup)
    logger.info("Shutting down app")

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# Include Routers
app.include_router(tubemood_router)
app.include_router(analytics_router)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to TubeMood API."}

