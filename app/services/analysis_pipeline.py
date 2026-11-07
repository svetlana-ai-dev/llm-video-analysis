"""Shared orchestration; analysis algorithms live in their existing services."""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import HTTPException, Response

from app.models.schemas import AnalysisResponse, LLMResult, MeetingSummaryResult, SpeechMetrics
from app.services import AudioAnalyzer, LLMAnalyzer, VideoAnalyzer, save_analysis_result
from app.services.meeting_summary import MeetingSummaryAnalyzer


def analyze_path(video_path: Path, filename: str, response: Response, started: float) -> AnalysisResponse:
    try:
        video = VideoAnalyzer(float(os.getenv("VIDEO_SAMPLE_FPS", "5"))).analyze(video_path)
    except Exception as exc:
        raise HTTPException(422, f"Не удалось проанализировать видео: {exc}") from exc

    wav_path = video_path.parent / "audio.wav"
    audio_analyzer = AudioAnalyzer()
    try:
        audio_analyzer.extract_audio(video_path, wav_path)
        speech = audio_analyzer.analyze(wav_path)
    except Exception as exc:
        speech = SpeechMetrics(status="failed", transcription_model=audio_analyzer.model, warnings=[str(exc)], observations=["Аудиодорожка не обработана."])
    try:
        meeting_summary = MeetingSummaryAnalyzer().generate(speech)
    except Exception:
        meeting_summary = MeetingSummaryResult(status="failed", message="Не удалось сформировать конспект встречи.")
    if speech.status == "failed":
        llm = LLMResult(status="unavailable", message="LLM-анализ не выполнялся из-за ошибки транскрипции.")
    else:
        llm = LLMAnalyzer().analyze(video, speech)
    result = AnalysisResponse(filename=filename, video=video, speech=speech, llm=llm, meeting_summary=meeting_summary, processing_seconds=round(time.perf_counter() - started, 3))
    try:
        save_analysis_result(result, video_path)
    except Exception:
        # Preserve the completed result even when the archive cannot be written.
        result.saved_result_path = None
        response.headers["X-Archive-Status"] = "failed"
    else:
        response.headers["X-Archive-Status"] = "saved"
    return result
