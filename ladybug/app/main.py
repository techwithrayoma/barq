from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .routes import training

# ---------- Configure global logging ----------

app = FastAPI()

# Include routers
app.include_router(training.router)

@app.get("/")
async def health():
    """
    Health check
    """
    return JSONResponse(
        status_code=200,
        content={"status": "LadyBug ✿ is healthy"}
    )
