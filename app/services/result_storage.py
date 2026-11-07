from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.schemas import AnalysisResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


def save_analysis_result(result: AnalysisResponse, video_path: Path | None = None) -> Path:
    """Persist one immutable analysis version and return its directory."""
    root = Path(os.getenv("RESULTS_DIR", str(DEFAULT_RESULTS_ROOT))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    version_dir = _create_version_directory(root, result.filename)
    staging = root / f".pending-{uuid.uuid4()}"
    previous_path = result.saved_result_path
    try:
        staging.mkdir()
        result.saved_result_path = str(version_dir)
        data = result.model_dump(mode="json")
        source_name = None
        if video_path is not None:
            source_name = f"source{video_path.suffix.lower()}"
            shutil.copyfile(video_path, staging / source_name)
        _write_text_exclusive(staging / "result.json", json.dumps(data, ensure_ascii=False, indent=2))
        _write_text_exclusive(staging / "summary.md", _render_summary(result))
        _save_timebase_artifacts(staging, data)
        metadata = {
            "id": str(uuid.uuid4()), "filename": result.filename,
            "analyzed_at": datetime.now().astimezone().isoformat(),
            "source_file": source_name,
            "source_size": (staging / source_name).stat().st_size if source_name else None,
            "storage_version": 1,
        }
        _write_text_exclusive(staging / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        version_dir.rmdir()
        staging.rename(version_dir)
        return version_dir
    except Exception:
        result.saved_result_path = previous_path
        if staging.exists():
            shutil.rmtree(staging)
        if version_dir.exists() and not any(version_dir.iterdir()):
            version_dir.rmdir()
        raise


def _archive_records():
    root = Path(os.getenv("RESULTS_DIR", str(DEFAULT_RESULTS_ROOT))).resolve()
    if not root.is_dir():
        return
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir() or directory.is_symlink() or directory.name.startswith("."):
            continue
        try:
            result_path = directory / "result.json"
            if result_path.is_symlink():
                continue
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not all(k in data for k in ("filename", "video", "speech", "llm")):
                continue
            metadata_path = directory / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() and not metadata_path.is_symlink() else {}
            record_id = metadata.get("id") or "legacy-" + hashlib.sha256(directory.name.encode()).hexdigest()
            source_name = metadata.get("source_file")
            source = directory / source_name if isinstance(source_name, str) and source_name == Path(source_name).name and source_name.startswith("source.") else None
            if source is not None and (not source.is_file() or source.is_symlink() or source.resolve().parent != directory.resolve()):
                source = None
            item = {
                "id": record_id, "filename": data["filename"],
                "analyzed_at": metadata.get("analyzed_at"),
                "video_available": source is not None,
                "video_url": f"/api/archive/{record_id}/video" if source else None,
                "warning": None if source else "Исходное видео не сохранено. Доступен полный сохранённый анализ.",
            }
            yield item, data, source
        except (OSError, ValueError, TypeError, AttributeError):
            continue


def list_archive() -> list[dict]:
    return [item for item, _, _ in _archive_records()]


def load_archive(record_id: str):
    for item, data, source in _archive_records():
        if item["id"] == record_id:
            return item, data, source
    raise FileNotFoundError("Сохранённая запись не найдена")


def _create_version_directory(root: Path, filename: str) -> Path:
    stem = Path(filename).stem or "video"
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" ._") or "video"
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    base_name = f"{safe_stem}_{timestamp}"
    for suffix in range(1000):
        candidate = root / (base_name if suffix == 0 else f"{base_name}_{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Не удалось создать уникальную папку версии результата")


def _write_text_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        output.write(content)


def _render_summary(result: AnalysisResponse) -> str:
    speech = result.speech
    qa = speech.qa_segmentation
    lines = [
        f"# Результат анализа: {result.filename}",
        "",
        f"- Сохранено в: `{result.saved_result_path}`",
        f"- Время обработки: {result.processing_seconds} с",
        f"- Статус видеоанализа: {result.video.status}",
        f"- Статус транскрипции: {speech.status}",
        f"- Модель транскрипции: {speech.transcription_model or 'не указана'}",
        f"- Статус Q&A-сегментации: {qa.status}",
        f"- Статус LLM-анализа: {result.llm.status}",
        "",
        "## Транскрипция",
        "",
        speech.transcript or "Транскрипция отсутствует.",
        "",
    ]
    lines.extend(["## Краткое содержание встречи", ""])
    summary = result.meeting_summary
    if summary is None:
        lines.append("Конспект для этой записи не сформирован.")
    elif summary.status != "completed" or summary.content is None:
        lines.append(summary.message or "Конспект недоступен.")
    else:
        content = summary.content
        lines.extend([" ".join(content.summary_sentences), "", "### Принятые решения", ""])
        lines.extend([f"- {item}" for item in content.decisions] or ["Не выявлены"])
        lines.extend(["", "### Поставленные задачи", ""])
        if not content.tasks:
            lines.append("Не выявлены")
        for task in content.tasks:
            details = []
            if task.assignee:
                details.append(f"Ответственный: {task.assignee}")
            if task.deadline:
                details.append(f"Срок: {task.deadline}")
            lines.append(f"- {task.task}" + (" — " + "; ".join(details) if details else ""))
        lines.extend(["", "### Открытые вопросы", ""])
        lines.extend([f"- {item}" for item in content.open_questions] or ["Не выявлены"])
    lines.extend(["", "## Вопросы и ответы по времени", ""])
    if not qa.blocks:
        lines.append(qa.message or "Смысловые пары не выделены.")
    for block in qa.blocks:
        lines.extend([
            f"### {block.index}. {block.title}",
            "",
            f"**Вопрос / задание ({block.question_start:.3f}–{block.question_end:.3f} с):** {block.question}",
            "",
            f"**Ответ ({block.answer_start:.3f}–{block.answer_end:.3f} с):** {block.answer}",
            "",
            f"Задержка перед ответом: {block.response_delay_seconds:.3f} с.",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _save_timebase_artifacts(version_dir: Path, data: dict[str, Any]) -> None:
    timebase = _find_timebase_mapping(data)
    if timebase is None:
        return
    _write_text_exclusive(version_dir / "timebase_mapping.json", json.dumps(timebase, ensure_ascii=False, indent=2))
    rows = timebase if isinstance(timebase, list) else timebase.get("mappings") if isinstance(timebase, dict) else None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (version_dir / "timebase_mapping.csv").open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _find_timebase_mapping(value: Any) -> Any | None:
    if isinstance(value, dict):
        for key in ("timebase_mapping", "timebase_mappings"):
            if key in value and value[key] is not None:
                return value[key]
        for nested in value.values():
            found = _find_timebase_mapping(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_timebase_mapping(nested)
            if found is not None:
                return found
    return None
