from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from openai import APITimeoutError, OpenAI

from app.models.schemas import SpeechMetrics, TranscriptSegment
from app.services.interview_segmentation import InterviewSegmenter


class AudioAnalyzer:
    """Extracts audio, computes signal-level speech metrics and optionally transcribes it."""

    def __init__(self, api_key: str | None = None, transcription_model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = transcription_model or os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")

    def extract_audio(self, video_path: str | Path, wav_path: str | Path) -> None:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(wav_path)]
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                process.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
        if process.returncode != 0:
            raise ValueError("В видео не найден пригодный для обработки аудиопоток")

    def analyze(self, wav_path: str | Path) -> SpeechMetrics:
        signal = self._signal_metrics(wav_path)
        if not self.api_key:
            return SpeechMetrics(status="unavailable", transcription_model=self.model, **signal, observations=self._observations(signal), warnings=["OPENAI_API_KEY не задан: транскрипция и производные текстовые метрики недоступны."])
        try:
            client = OpenAI(api_key=self.api_key)
            with open(wav_path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model=self.model, file=audio, response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            text = transcript.text or ""
            raw_segments = getattr(transcript, "segments", None) or []
            segments = [TranscriptSegment(start=getattr(s, "start", None), end=getattr(s, "end", None), text=getattr(s, "text", "").strip()) for s in raw_segments]
            words = len(text.split())
            duration = signal["duration_seconds"]
            wpm = words / duration * 60 if duration else None
            observations = self._observations(signal)
            if wpm is not None:
                observations.append(f"Расчётный темп по транскрипту: {wpm:.0f} слов/мин.")
            qa_segmentation = InterviewSegmenter(api_key=self.api_key).segment(segments)
            return SpeechMetrics(status="completed", transcription_model=self.model, transcript=text, segments=segments, qa_segmentation=qa_segmentation, words_count=words, estimated_words_per_minute=round(wpm, 1) if wpm else None, observations=observations, **signal)
        except APITimeoutError as exc:
            cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "UnknownTimeout"
            return SpeechMetrics(
                status="failed",
                transcription_model=self.model,
                **signal,
                observations=self._observations(signal),
                warnings=[
                    "Транскрипция не выполнена: Request timed out; "
                    f"этап=transcription, причина={cause}."
                ],
            )
        except Exception as exc:
            return SpeechMetrics(status="failed", transcription_model=self.model, **signal, observations=self._observations(signal), warnings=[f"Транскрипция не выполнена: {exc}"])

    @staticmethod
    def _signal_metrics(wav_path: str | Path) -> dict:
        with wave.open(str(wav_path), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
        duration = len(samples) / rate if rate else 0
        window = max(1, int(rate * 0.03))
        chunks = samples[: len(samples) // window * window].reshape(-1, window) if len(samples) >= window else samples.reshape(1, -1)
        rms = np.sqrt(np.mean(chunks ** 2, axis=1) + 1e-12)
        adaptive_threshold = max(0.008, float(np.percentile(rms, 25) * 2.2))
        voiced = rms > adaptive_threshold
        pause_count = int(np.sum((~voiced[1:]) & voiced[:-1])) if len(voiced) > 1 else 0
        voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0
        return {"duration_seconds": round(duration, 3), "voiced_ratio": round(voiced_ratio, 4), "pause_ratio": round(1 - voiced_ratio, 4), "estimated_pause_count": pause_count, "mean_rms": round(float(np.mean(rms)), 5)}

    @staticmethod
    def _observations(metrics: dict) -> list[str]:
        return [
            f"Доля окон с сигналом выше адаптивного порога: {metrics['voiced_ratio']:.1%}.",
            f"Доля низкоэнергетических окон: {metrics['pause_ratio']:.1%}. Они не равнозначны смысловым или психологическим паузам.",
            f"Оценочное число переходов к низкоэнергетическим окнам: {metrics['estimated_pause_count']}.",
        ]
