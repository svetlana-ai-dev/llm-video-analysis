from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Response
from fastapi.responses import FileResponse
from app.services.result_storage import list_archive, load_archive

from pydantic import BaseModel, Field

from app.models.schemas import AnalysisResponse
from app.services.analysis_pipeline import analyze_path
from app.services.cloud_download import DownloadError, download_video

router = APIRouter(prefix="/api")
ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_video(response: Response, file: UploadFile = File(...)) -> AnalysisResponse:
    started = time.perf_counter()
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Поддерживаются MP4, MOV, WebM, MKV, AVI и M4V")
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="llm_video_") as temp_dir:
        video_path = Path(temp_dir) / f"input{suffix}"
        size = 0
        with video_path.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Файл превышает лимит {max_bytes // 1024 // 1024} МБ")
                target.write(chunk)
        return analyze_path(video_path, file.filename or "video", response, started)


class CloudVideoInput(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


@router.post("/analyze-url", response_model=AnalysisResponse)
def analyze_cloud_video(data: CloudVideoInput, response: Response) -> AnalysisResponse:
    started = time.perf_counter()
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="llm_cloud_") as temp_dir:
        try:
            downloaded = download_video(data.url, Path(temp_dir), max_bytes)
        except DownloadError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        return analyze_path(downloaded.path, downloaded.filename, response, started)


@router.get("/archive")
def archive_list():
    return list_archive()


def _get_record(record_id: str):
    try:
        return load_archive(record_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/archive/{record_id}")
def archive_detail(record_id: str):
    metadata, result, _ = _get_record(record_id)
    return {"metadata": metadata, "result": result}


@router.get("/archive/{record_id}/video")
def archive_video(record_id: str):
    _, _, source = _get_record(record_id)
    if source is None:
        raise HTTPException(404, "Исходное видео не сохранено")
    return FileResponse(source)
