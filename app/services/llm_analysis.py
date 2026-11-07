from __future__ import annotations

import json
import os
import re

from openai import APITimeoutError, OpenAI

from app.models.schemas import LLMAnalysis, LLMResult, SpeechMetrics, VideoMetrics


SYSTEM_PROMPT = """Ты анализируешь только наблюдаемые и алгоритмически измеренные признаки видео и речи. Разрешены констатации вида: «увеличилась амплитуда движения головы», «зафиксирована улыбкоподобная геометрия», «изменился расчётный темп речи», и осторожное временное сопоставление таких сигналов.

Запрещено делать прямые или косвенные выводы о правдивости/лжи, истинности слов, внутренних эмоциях, психологическом состоянии, уверенности, подготовленности, личностных качествах, психотипе или диагнозах. Не оценивай кандидата и не принимай кадровые решения. Не называй низкоэнергетические окна смысловыми или психологическими паузами: это только уровень энергии сигнала. В каждом поле, включая interpretation, используй только эти рамки. Вне limitations описывай уровень энергии как «низкоэнергетические окна», не употребляй слово «пауза». Без предоставленных нормативов не называй числовые показатели аномальными, повышенными или пониженными. Baseline — исходный уровень калибровки этого видео; изменение относительно него — не отклонение от нормы. Очки, частично недоступный глаз и низкое качество распознавания — ограничения измерений, а не признаки поведения человека. Учитывай ограничения ракурса, освещения, качества звука и алгоритмов. Пиши по-русски."""

DISALLOWED_INFERENCE = re.compile(
    r"\b(правдив|истинн|ложь|лжи|лжив|эмоци|психолог|уверенн|подготовленн|личностн)\w*",
    re.IGNORECASE,
)
# Accept an explicit acoustic label, never an inferred speech pause.
LOW_ENERGY_PAUSE = re.compile(r"пауз\w*\s+с\s+низкой\s+энергией\b", re.IGNORECASE)
PAUSE_TERM = re.compile(r"\bпауз\w*", re.IGNORECASE)
TECHNICAL_RECOGNITION_CONTEXT = re.compile(
    r"(?:качест\w*|над[её]жност\w*|техническ\w*|алгоритм\w*)"
    r".{0,60}(?:распознаван\w*|определен\w*|детекц\w*|глаз\w*|eye[ -]?tracking)",
    re.IGNORECASE,
)


def _analysis_validation_issues(analysis: LLMAnalysis) -> list[dict[str, str]]:
    """Return compact field-level diagnostics without retaining the full answer."""
    issues: list[dict[str, str]] = []
    data = analysis.model_dump(exclude={"limitations"})

    def inspect(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                inspect(nested, f"{path}.{key}" if path else key)
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                inspect(nested, f"{path}[{index}]")
            return
        if not isinstance(value, str):
            return
        for pattern in (DISALLOWED_INFERENCE, PAUSE_TERM):
            for match in pattern.finditer(value):
                if pattern is PAUSE_TERM and LOW_ENERGY_PAUSE.match(value, match.start()):
                    continue
                start, end = max(0, match.start() - 55), min(len(value), match.end() + 55)
                context = value[start:end].replace("\n", " ").strip()
                # Technical statements about recognition quality/reliability are
                # measurements of the algorithm, not claims about a person.
                if match.group(0).lower().startswith("уверенн") and TECHNICAL_RECOGNITION_CONTEXT.search(context):
                    continue
                issues.append({"field": path, "match": match.group(0), "context": context})
    inspect(data, "")
    return issues


def _analysis_requires_rewrite(analysis: LLMAnalysis) -> bool:
    return bool(_analysis_validation_issues(analysis))


def _build_llm_payload(video: VideoMetrics, speech: SpeechMetrics) -> dict:
    """Expose objective values without duplicating prewritten interpretive prose."""
    video_data = video.model_dump(mode="json")
    speech_data = speech.model_dump(mode="json")
    speech_data.pop("observations", None)
    pause_ratio = speech_data.pop("pause_ratio", None)
    estimated_pause_count = speech_data.pop("estimated_pause_count", None)
    speech_data["low_energy_ratio"] = pause_ratio
    speech_data["estimated_low_energy_window_count"] = estimated_pause_count

    eye = video_data.get("eye_tracking")
    if isinstance(eye, dict):
        left = eye.pop("mean_left_eye_confidence", None)
        right = eye.pop("mean_right_eye_confidence", None)
        eye["mean_left_eye_recognition_quality"] = left
        eye["mean_right_eye_recognition_quality"] = right
        sources = eye.get("gaze_source_distribution")
        if isinstance(sources, dict) and "low_eye_confidence" in sources:
            sources["low_eye_recognition_quality"] = sources.pop("low_eye_confidence")
        eye["recognition_quality_note"] = (
            "Показатели от 0 до 1 описывают качество распознавания глаза алгоритмом, "
            "а не личные характеристики или состояние человека."
        )
    return {"video_observations": video_data, "speech_observations": speech_data}


class LLMAnalyzer:
    """Independent analytical layer consuming normalized video and speech metrics."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def analyze(self, video: VideoMetrics, speech: SpeechMetrics) -> LLMResult:
        if not self.api_key:
            return LLMResult(status="unavailable", model=self.model, message="OPENAI_API_KEY не задан. Измеренные признаки доступны, LLM-анализ отключён.")
        payload = _build_llm_payload(video, speech)
        try:
            client = OpenAI(api_key=self.api_key)
            user_content = "Сформируй структурированный анализ этих данных:\n" + json.dumps(payload, ensure_ascii=False)
            parsed = None
            issues: list[dict[str, str]] = []
            attempt_diagnostics: list[dict[str, str | int]] = []
            for attempt in range(2):
                response = client.responses.parse(
                    model=self.model,
                    input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
                    text_format=LLMAnalysis,
                )
                parsed = response.output_parsed
                issues = _analysis_validation_issues(parsed) if parsed is not None else []
                if parsed is not None and not issues:
                    break
                attempt_diagnostics.extend(
                    {"attempt": attempt + 1, **issue}
                    for issue in issues[:3]
                )
                user_content = (
                    "Исправь предыдущий структурированный ответ по всем замечаниям валидатора. "
                    "Сохрани остальные наблюдаемые факты. Удали выводы об эмоциях и внутренних состояниях; "
                    "не заменяй их синонимами. Уровень энергии описывай как низкоэнергетические окна, "
                    "без предположений о речевых паузах. Вне limitations не повторяй запрещённые "
                    "термины даже в отрицании. Верни полный исправленный структурированный ответ.\n"
                    "Замечания валидатора:\n" + json.dumps(issues, ensure_ascii=False)
                    + "\nПредыдущий ответ:\n"
                    + (parsed.model_dump_json() if parsed is not None else "Структурированный ответ отсутствует")
                    + "\nИсходные данные:\n" + json.dumps(payload, ensure_ascii=False)
                )
            if parsed is None:
                raise ValueError("Модель не вернула структурированный результат")
            issues = _analysis_validation_issues(parsed)
            if issues:
                diagnostic = "; ".join(
                    f"попытка={item['attempt']}, поле={item['field']}, "
                    f"совпадение={item['match']!r}, контекст={item['context']!r}"
                    for item in attempt_diagnostics
                )
                raise ValueError(
                    "LLM-анализ отклонён: текст вышел за рамки наблюдаемых фактов; "
                    + diagnostic
                )
            return LLMResult(status="completed", model=self.model, analysis=parsed)
        except APITimeoutError as exc:
            cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "UnknownTimeout"
            return LLMResult(
                status="failed",
                model=self.model,
                message=(
                    "LLM-анализ не выполнен: Request timed out; "
                    f"этап=llm_response_attempt_{attempt + 1}, причина={cause}."
                ),
            )
        except Exception as exc:
            return LLMResult(status="failed", model=self.model, message=f"LLM-анализ не выполнен: {exc}")
