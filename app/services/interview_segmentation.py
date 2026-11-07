from __future__ import annotations

import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from app.models.schemas import (
    InterviewQABlock,
    InterviewQASegmentation,
    TranscriptSegment,
)


SYSTEM_PROMPT = """Ты размечаешь интервью на смысловые пары «вопрос/задание -> ответ».
Новый блок начинается не только с вопросительного предложения, но и с инструкции,
кейса, исходных условий задачи или существенного изменения ранее заданных условий.
Не дели текст механически по знакам вопроса. Объединяй соседние реплики интервьюера,
относящиеся к одному заданию. Ответом считай следующие сегменты кандидата до начала
следующего смыслового задания. Возвращай только диапазоны ID входных сегментов.
Не придумывай текст, роли говорящих или временные метки. Если уверенно выделить пару
«задание -> ответ» нельзя, не создавай блок. Диапазоны должны идти по времени, не
пересекаться и содержать хотя бы по одному сегменту вопроса и ответа. Пиши title по-русски."""


class _ProposedBlock(BaseModel):
    type: Literal["question", "instruction", "case", "condition_change"]
    title: str
    question_segment_ids: list[int] = Field(min_length=1)
    answer_segment_ids: list[int] = Field(min_length=1)


class _ProposedSegmentation(BaseModel):
    blocks: list[_ProposedBlock] = Field(default_factory=list)


class InterviewSegmenter:
    """Maps semantic Q&A boundaries to immutable transcription segment timestamps."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv(
            "OPENAI_SEGMENTATION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        )

    def segment(self, segments: list[TranscriptSegment]) -> InterviewQASegmentation:
        usable = [segment for segment in segments if segment.start is not None and segment.end is not None and segment.text.strip()]
        if not self.api_key:
            return InterviewQASegmentation(status="unavailable", model=self.model, message="OPENAI_API_KEY не задан: смысловая Q&A-сегментация недоступна.")
        if len(usable) < 2:
            return InterviewQASegmentation(status="completed", model=self.model, blocks=[], message="Недостаточно timestamp-сегментов для выделения пары вопрос–ответ.")

        indexed = [
            {"id": index, "start": segment.start, "end": segment.end, "text": segment.text}
            for index, segment in enumerate(segments)
            if segment.start is not None and segment.end is not None and segment.text.strip()
        ]
        try:
            response = OpenAI(api_key=self.api_key).responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Разметь timestamp-сегменты интервью:\n" + json.dumps(indexed, ensure_ascii=False)},
                ],
                text_format=_ProposedSegmentation,
            )
            proposed = response.output_parsed
            if proposed is None:
                raise ValueError("модель не вернула структурированную разметку")
            blocks = self._materialize(proposed, segments)
            return InterviewQASegmentation(status="completed", model=self.model, blocks=blocks)
        except Exception as exc:
            return InterviewQASegmentation(status="failed", model=self.model, message=f"Q&A-сегментация не выполнена: {exc}")

    @staticmethod
    def _materialize(proposed: _ProposedSegmentation, segments: list[TranscriptSegment]) -> list[InterviewQABlock]:
        normalized: list[tuple[_ProposedBlock, int, int, int, int]] = []
        for item in proposed.blocks:
            question_ids = sorted({index for index in item.question_segment_ids if 0 <= index < len(segments)})
            answer_ids = sorted({index for index in item.answer_segment_ids if 0 <= index < len(segments)})
            if not question_ids or not answer_ids:
                continue

            question_start_id, question_end_id = question_ids[0], question_ids[-1]
            answer_start_id, answer_end_id = answer_ids[0], answer_ids[-1]

            # Treat model output as ranges. If their edges overlap, put the shared
            # source segment on one side of a real segment boundary.
            if question_end_id >= answer_start_id:
                if question_start_id < answer_start_id:
                    question_end_id = answer_start_id - 1
                elif question_start_id < answer_end_id:
                    question_end_id = question_start_id
                    answer_start_id = question_start_id + 1
                else:
                    continue

            if question_start_id > question_end_id or answer_start_id > answer_end_id:
                continue
            normalized.append((item, question_start_id, question_end_id, answer_start_id, answer_end_id))

        normalized.sort(key=lambda value: (value[1], value[3]))
        resolved: list[tuple[_ProposedBlock, int, int, int, int]] = []
        for candidate in normalized:
            item, question_start_id, question_end_id, answer_start_id, answer_end_id = candidate
            if resolved and question_start_id <= resolved[-1][4]:
                previous = resolved[-1]
                trimmed_previous_answer_end = question_start_id - 1
                if trimmed_previous_answer_end >= previous[3]:
                    resolved[-1] = (*previous[:4], trimmed_previous_answer_end)
                else:
                    question_start_id = previous[4] + 1
                    if question_start_id > question_end_id:
                        continue
            resolved.append((item, question_start_id, question_end_id, answer_start_id, answer_end_id))

        blocks: list[InterviewQABlock] = []
        for item, question_start_id, question_end_id, answer_start_id, answer_end_id in resolved:
            question_ids = list(range(question_start_id, question_end_id + 1))
            answer_ids = list(range(answer_start_id, answer_end_id + 1))
            selected = [segments[index] for index in question_ids + answer_ids]
            if any(segment.start is None or segment.end is None or not segment.text.strip() for segment in selected):
                continue

            question_start = float(segments[question_start_id].start)
            raw_question_end = float(segments[question_end_id].end)
            answer_start = float(segments[answer_start_id].start)
            answer_end = float(segments[answer_end_id].end)
            if question_start > answer_start or answer_start > answer_end:
                continue

            # Whisper segments can have slightly overlapping times. Use the next
            # segment's real start boundary; never synthesize a timestamp.
            question_end = min(raw_question_end, answer_start)
            if question_start > question_end:
                continue
            blocks.append(
                InterviewQABlock(
                    index=len(blocks) + 1,
                    type=item.type,
                    title=item.title.strip() or f"Блок {len(blocks) + 1}",
                    question=" ".join(segments[index].text.strip() for index in question_ids),
                    question_start=question_start,
                    question_end=question_end,
                    answer=" ".join(segments[index].text.strip() for index in answer_ids),
                    answer_start=answer_start,
                    answer_end=answer_end,
                    response_delay_seconds=round(max(0.0, answer_start - question_end), 3),
                    question_segment_ids=question_ids,
                    answer_segment_ids=answer_ids,
                )
            )
        return blocks
