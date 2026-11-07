from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    time: float
    end_time: float | None = None
    type: str
    description: str
    value: float | str | None = None


class VideoCalibration(BaseModel):
    completed: bool = False
    duration_seconds: float = 0
    samples: int = 0
    baseline: dict[str, float] = Field(default_factory=dict)
    message: str = "Калибровка не выполнена."


class EyeTrackingQuality(BaseModel):
    """Additive diagnostics for eye/gaze measurements."""

    status: Literal["reliable", "partial", "unavailable"] = "unavailable"
    reliable_frames: int = 0
    reliable_ratio: float = 0
    both_eyes_frames: int = 0
    single_eye_frames: int = 0
    unavailable_frames: int = 0
    mode_distribution: dict[str, float] = Field(default_factory=dict)
    gaze_source_distribution: dict[str, float] = Field(default_factory=dict)
    mean_left_eye_confidence: float | None = None
    mean_right_eye_confidence: float | None = None
    left_quality_reasons: dict[str, int] = Field(default_factory=dict)
    right_quality_reasons: dict[str, int] = Field(default_factory=dict)
    limitation_reason: str | None = None


class VideoMetrics(BaseModel):
    status: Literal["completed", "unavailable", "failed"] = "completed"
    analyzer: str
    duration_seconds: float = 0
    source_fps: float = 0
    analyzed_fps: float = 0
    frames_total: int = 0
    frames_analyzed: int = 0
    frames_with_face: int = 0
    face_presence_ratio: float = 0
    blink_count: int = 0
    eyes_closed_seconds: float = 0
    gaze_distribution: dict[str, float] = Field(default_factory=dict)
    gaze_changes: int = 0
    smile_ratio: float = 0
    eyebrow_movement_mean: float = 0
    mouth_open_ratio: float = 0
    facial_change_mean: float = 0
    head_pose_mean: dict[str, float] = Field(default_factory=dict)
    head_movement_mean: float = 0
    calibration: VideoCalibration = Field(default_factory=VideoCalibration)
    eye_tracking: EyeTrackingQuality = Field(default_factory=EyeTrackingQuality)
    events: list[TimelineEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str


class InterviewQABlock(BaseModel):
    index: int
    type: Literal["question", "instruction", "case", "condition_change"]
    title: str
    question: str
    question_start: float
    question_end: float
    answer: str
    answer_start: float
    answer_end: float
    response_delay_seconds: float
    question_segment_ids: list[int] = Field(default_factory=list)
    answer_segment_ids: list[int] = Field(default_factory=list)


class InterviewQASegmentation(BaseModel):
    status: Literal["completed", "unavailable", "failed", "not_requested"] = "not_requested"
    model: str | None = None
    blocks: list[InterviewQABlock] = Field(default_factory=list)
    message: str | None = None


class SpeechMetrics(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    transcription_model: str | None = None
    transcript: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    qa_segmentation: InterviewQASegmentation = Field(default_factory=InterviewQASegmentation)
    duration_seconds: float = 0
    voiced_ratio: float | None = None
    pause_ratio: float | None = None
    estimated_pause_count: int | None = None
    words_count: int = 0
    estimated_words_per_minute: float | None = None
    mean_rms: float | None = None
    observations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LLMAnalysis(BaseModel):
    summary: str
    facial_analysis: str
    gaze_analysis: str
    head_movement_analysis: str
    speech_analysis: str
    cross_modal_analysis: str
    timeline_insights: list[str]
    observed_facts: list[str]
    interpretation: list[str]
    limitations: list[str]


class LLMResult(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    model: str | None = None
    analysis: LLMAnalysis | None = None
    message: str | None = None


class MeetingTask(BaseModel):
    task: str
    assignee: str | None = None
    deadline: str | None = None
    source_quote: str


class MeetingSummaryContent(BaseModel):
    summary_sentences: list[str] = Field(min_length=5, max_length=8)
    decisions: list[str] = Field(default_factory=list)
    tasks: list[MeetingTask] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class MeetingSummaryResult(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    model: str | None = None
    content: MeetingSummaryContent | None = None
    message: str | None = None


class AnalysisResponse(BaseModel):
    filename: str
    saved_result_path: str | None = None
    video: VideoMetrics
    speech: SpeechMetrics
    llm: LLMResult
    meeting_summary: MeetingSummaryResult | None = None
    processing_seconds: float
    objective_data_note: str = "Поля video и speech содержат измеренные или алгоритмически рассчитанные признаки."
    interpretation_note: str = "Поля llm.analysis содержат интерпретацию модели и не являются диагнозом, оценкой психотипа или определением лжи."


def model_to_jsonable(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
