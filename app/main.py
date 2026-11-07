from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.auth import install_auth

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Interview Behavior Video Analysis API",
    description="Измерение наблюдаемых мимических, глазодвигательных, двигательных и речевых признаков с отдельным LLM-слоем.",
    version="0.2.0",
)
app.include_router(router)
install_auth(app)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health", tags=["system"])
def health() -> dict:
    capabilities = {"video": True, "transcription": bool(os.getenv("OPENAI_API_KEY")), "llm": bool(os.getenv("OPENAI_API_KEY"))}
    try:
        import mediapipe
        capabilities["mediapipe"] = hasattr(mediapipe, "solutions")
    except Exception:
        capabilities["mediapipe"] = False
    return {"status": "ok", "version": app.version, "capabilities": capabilities, "models": {"llm": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "transcription": os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")}}
