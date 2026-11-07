from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient
from openai import APITimeoutError

from app.main import app
from app.models.schemas import LLMAnalysis, LLMResult, SpeechMetrics, TimelineEvent, TranscriptSegment, VideoMetrics
from app.services.interview_segmentation import InterviewSegmenter, _ProposedBlock, _ProposedSegmentation
from app.services.audio_analysis import AudioAnalyzer
from app.services.llm_analysis import (
    LLMAnalyzer,
    _analysis_requires_rewrite,
    _analysis_validation_issues,
    _build_llm_payload,
)
from app.services.video_analysis import (
    EyeMeasurement,
    VideoAnalyzer,
    _DebouncedState,
    _classify_gaze,
    _estimate_gaze,
    _merge_report_events,
    _robust_median,
)


client = TestClient(app)


def _llm_analysis(**overrides) -> LLMAnalysis:
    values = {
        "summary": "Зафиксированы измеренные изменения.",
        "facial_analysis": "Измерена геометрия лица.",
        "gaze_analysis": "Направление взгляда рассчитано по пригодным кадрам.",
        "head_movement_analysis": "Измерены углы положения головы.",
        "speech_analysis": "Измерены характеристики аудиосигнала.",
        "cross_modal_analysis": "Сопоставлены временные измерения.",
        "timeline_insights": [],
        "observed_facts": [],
        "interpretation": [],
        "limitations": [],
    }
    values.update(overrides)
    return LLMAnalysis(**values)


def test_llm_validator_allows_technical_eye_recognition_quality():
    analysis = _llm_analysis(
        gaze_analysis="Техническая уверенность распознавания глаза алгоритмом составила 0,74."
    )

    assert _analysis_requires_rewrite(analysis) is False
    assert _analysis_validation_issues(analysis) == []


def test_llm_validator_rejects_person_confidence_with_field_diagnostic():
    analysis = _llm_analysis(interpretation=["Кандидат демонстрирует уверенность человека в ответе."])

    issues = _analysis_validation_issues(analysis)
    assert _analysis_requires_rewrite(analysis) is True
    assert issues[0]["field"] == "interpretation[0]"
    assert issues[0]["match"].lower().startswith("уверенн")
    assert "Кандидат" in issues[0]["context"]


def test_llm_payload_omits_observation_prose_and_renames_eye_confidence():
    video = VideoMetrics(
        analyzer="test",
        eye_tracking={
            "status": "partial",
            "mean_left_eye_confidence": 0.74,
            "mean_right_eye_confidence": 0.71,
            "gaze_source_distribution": {"low_eye_confidence": 0.1},
        },
    )
    speech = SpeechMetrics(
        status="completed",
        pause_ratio=0.42,
        estimated_pause_count=17,
        observations=["Окна не равнозначны психологическим паузам."],
    )

    payload = _build_llm_payload(video, speech)
    serialized = str(payload)
    eye = payload["video_observations"]["eye_tracking"]
    assert "observations" not in payload["speech_observations"]
    assert "pause_ratio" not in payload["speech_observations"]
    assert "estimated_pause_count" not in payload["speech_observations"]
    assert payload["speech_observations"]["low_energy_ratio"] == 0.42
    assert payload["speech_observations"]["estimated_low_energy_window_count"] == 17
    assert "психологическим паузам" not in serialized
    assert "mean_left_eye_confidence" not in eye
    assert eye["mean_left_eye_recognition_quality"] == 0.74
    assert "low_eye_recognition_quality" in eye["gaze_source_distribution"]


def test_llm_validator_still_rejects_psychological_inference():
    analysis = _llm_analysis(summary="Эти движения показывают психологическое состояние человека.")

    issues = _analysis_validation_issues(analysis)
    assert _analysis_requires_rewrite(analysis) is True
    assert issues[0]["field"] == "summary"
    assert issues[0]["match"].lower().startswith("психолог")


def test_llm_rejection_reports_compact_diagnostics_for_both_attempts(monkeypatch):
    responses = iter(
        [
            _llm_analysis(summary="Это психологическая интерпретация."),
            _llm_analysis(speech_analysis="Обнаружены паузы в речи."),
        ]
    )

    class FakeResponses:
        def parse(self, **kwargs):
            return type("Response", (), {"output_parsed": next(responses)})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr("app.services.llm_analysis.OpenAI", FakeOpenAI)

    result = LLMAnalyzer(api_key="test-key").analyze(
        VideoMetrics(analyzer="test"),
        SpeechMetrics(status="completed"),
    )

    assert result.status == "failed"
    assert "попытка=1, поле=summary" in result.message
    assert "совпадение='психологическая'" in result.message
    assert "попытка=2, поле=speech_analysis" in result.message
    assert "совпадение='паузы'" in result.message
    assert "полный ответ" not in result.message


def test_openai_timeout_diagnostics_include_stage_and_cause(monkeypatch, tmp_path):
    class TimeoutTranscriptions:
        def create(self, **kwargs):
            try:
                raise httpx.ConnectTimeout("connect")
            except httpx.ConnectTimeout as cause:
                raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid")) from cause

    class TimeoutAudio:
        transcriptions = TimeoutTranscriptions()

    class TimeoutResponses:
        def parse(self, **kwargs):
            try:
                raise httpx.ReadTimeout("read")
            except httpx.ReadTimeout as cause:
                raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid")) from cause

    class TimeoutOpenAI:
        def __init__(self, **kwargs):
            self.audio = TimeoutAudio()
            self.responses = TimeoutResponses()

    monkeypatch.setattr("app.services.audio_analysis.OpenAI", TimeoutOpenAI)
    monkeypatch.setattr("app.services.llm_analysis.OpenAI", TimeoutOpenAI)
    monkeypatch.setattr(
        AudioAnalyzer,
        "_signal_metrics",
        staticmethod(lambda path: {
            "duration_seconds": 1.0,
            "voiced_ratio": 0.5,
            "pause_ratio": 0.5,
            "estimated_pause_count": 1,
            "mean_rms": 0.1,
        }),
    )
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"wav")

    speech = AudioAnalyzer(api_key="test-key").analyze(wav_path)
    llm = LLMAnalyzer(api_key="test-key").analyze(
        VideoMetrics(analyzer="test"),
        SpeechMetrics(status="completed"),
    )

    assert speech.status == "failed"
    assert "этап=transcription" in speech.warnings[0]
    assert "причина=ConnectTimeout" in speech.warnings[0]
    assert llm.status == "failed"
    assert "этап=llm_response_attempt_1" in llm.message
    assert "причина=ReadTimeout" in llm.message


def test_failed_transcription_skips_llm(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(
        "app.services.analysis_pipeline.VideoAnalyzer.analyze",
        lambda self, path: VideoMetrics(analyzer="test", frames_analyzed=1),
    )
    monkeypatch.setattr(
        "app.services.analysis_pipeline.AudioAnalyzer.extract_audio",
        lambda self, source, target: Path(target).write_bytes(b"wav"),
    )
    monkeypatch.setattr(
        "app.services.analysis_pipeline.AudioAnalyzer.analyze",
        lambda self, path: SpeechMetrics(status="failed", transcription_model="whisper-test"),
    )

    def fail_if_called(self, video, speech):
        raise AssertionError("LLM must not be called after failed transcription")

    monkeypatch.setattr("app.services.analysis_pipeline.LLMAnalyzer.analyze", fail_if_called)

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["llm"] == {
        "status": "unavailable",
        "model": None,
        "analysis": None,
        "message": "LLM-анализ не выполнялся из-за ошибки транскрипции.",
    }


def test_health_and_interface():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "Добавить видео" in page.text
    assert "СРАВНЕНИЕ ВИДЕО" in page.text
    assert "Вопросы и ответы по времени" in page.text
    assert "Качество анализа" in page.text
    assert "Надёжность отслеживания глаз" in page.text
    assert "Технические детали измерений" in page.text
    assert "Сопоставление наблюдаемых данных" in page.text
    assert 'id="result-template"' in page.text
    assert client.get("/docs").status_code == 200


def test_analyze_contract(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr("app.services.analysis_pipeline.VideoAnalyzer.analyze", lambda self, path: VideoMetrics(analyzer="test", frames_analyzed=1))
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.extract_audio", lambda self, source, target: Path(target).write_bytes(b"wav"))
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.analyze", lambda self, path: SpeechMetrics(status="unavailable", transcription_model="whisper-test"))
    monkeypatch.setattr("app.services.analysis_pipeline.LLMAnalyzer.analyze", lambda self, video, speech: LLMResult(status="unavailable", message="no key"))
    response = client.post("/api/analyze", files={"file": ("sample.mp4", b"video-bytes", "video/mp4")})
    assert response.status_code == 200
    body = response.json()
    assert body["video"]["analyzer"] == "test"
    assert Path(body["saved_result_path"]).is_dir()
    assert (Path(body["saved_result_path"]) / "result.json").is_file()
    assert (Path(body["saved_result_path"]) / "summary.md").is_file()
    assert body["video"]["calibration"]["completed"] is False
    assert body["video"]["eye_tracking"]["status"] == "unavailable"
    assert body["speech"]["transcription_model"] == "whisper-test"
    assert body["speech"]["qa_segmentation"] == {
        "status": "not_requested",
        "model": None,
        "blocks": [],
        "message": None,
    }
    assert body["llm"]["status"] == "unavailable"


def test_repeated_filename_creates_immutable_versions(monkeypatch, tmp_path: Path):
    results_root = tmp_path / "results"
    monkeypatch.setenv("RESULTS_DIR", str(results_root))
    monkeypatch.setattr("app.services.analysis_pipeline.VideoAnalyzer.analyze", lambda self, path: VideoMetrics(analyzer="test", frames_analyzed=1))
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.extract_audio", lambda self, source, target: Path(target).write_bytes(b"wav"))
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.analyze", lambda self, path: SpeechMetrics(status="unavailable", transcription_model="whisper-test"))
    monkeypatch.setattr("app.services.analysis_pipeline.LLMAnalyzer.analyze", lambda self, video, speech: LLMResult(status="unavailable", message="no key"))

    first = client.post("/api/analyze", files={"file": ("same-video.mp4", b"first", "video/mp4")})
    assert first.status_code == 200
    first_dir = Path(first.json()["saved_result_path"])
    first_json_before = (first_dir / "result.json").read_bytes()
    first_summary_before = (first_dir / "summary.md").read_bytes()

    second = client.post("/api/analyze", files={"file": ("same-video.mp4", b"second", "video/mp4")})
    assert second.status_code == 200
    second_dir = Path(second.json()["saved_result_path"])

    assert first_dir != second_dir
    assert first_dir.name.startswith("same-video_")
    assert second_dir.name.startswith("same-video_")
    assert (second_dir / "result.json").is_file()
    assert (second_dir / "summary.md").is_file()
    assert (first_dir / "result.json").read_bytes() == first_json_before
    assert (first_dir / "summary.md").read_bytes() == first_summary_before
    assert len([path for path in results_root.iterdir() if path.is_dir()]) == 2


def test_copy_json_is_client_only_and_does_not_call_analysis_or_storage():
    script = (Path(__file__).parents[1] / "app" / "static" / "app.js").read_text(encoding="utf-8")
    copy_function = script.split("async function copyResult", 1)[1].split("function list", 1)[0]

    assert "navigator.clipboard.writeText" in copy_function
    assert "fetch(" not in copy_function
    assert "/api/analyze" not in copy_function
    assert "save" not in copy_function.lower()


def test_interface_maps_eye_tracking_and_quality_fields():
    script = (Path(__file__).parents[1] / "app" / "static" / "app.js").read_text(encoding="utf-8")

    required_fields = [
        "reliable_frames",
        "reliable_ratio",
        "both_eyes_frames",
        "single_eye_frames",
        "unavailable_frames",
        "mode_distribution",
        "gaze_source_distribution",
        "mean_left_eye_confidence",
        "mean_right_eye_confidence",
        "left_quality_reasons",
        "right_quality_reasons",
        "limitation_reason",
    ]
    for field in required_fields:
        assert field in script
    assert "очки" in script.lower()
    assert "не объявляет очки обнаруженным фактом" in script
    assert "Положение головы измеряется отдельно" in script


def test_interface_does_not_replace_missing_comparison_values_with_zero():
    script = (Path(__file__).parents[1] / "app" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'function metricValue(m,x){return x==null?"не определено"' in script
    assert "r.video.gaze_distribution?.center||0" not in script
    assert "r.speech.voiced_ratio??0" not in script


def test_qa_segmentation_uses_exact_source_timestamps_and_text():
    segments = [
        TranscriptSegment(start=10.2, end=14.8, text="Представьте, что до запуска пять дней."),
        TranscriptSegment(start=14.8, end=20.0, text="Как вы составите план?"),
        TranscriptSegment(start=20.7, end=28.4, text="Сначала я выделю критический путь."),
        TranscriptSegment(start=28.4, end=34.1, text="Затем распределю задачи по команде."),
        TranscriptSegment(start=35.0, end=41.3, text="Теперь осталось два дня, и один участник выбыл."),
        TranscriptSegment(start=42.0, end=49.6, text="Я пересмотрю объём и сразу сообщу о рисках."),
    ]
    proposed = _ProposedSegmentation(blocks=[
        _ProposedBlock(type="case", title="Мини-задача на планирование", question_segment_ids=[0, 1], answer_segment_ids=[2, 3]),
        _ProposedBlock(type="condition_change", title="Изменение условий", question_segment_ids=[4], answer_segment_ids=[5]),
    ])

    blocks = InterviewSegmenter._materialize(proposed, segments)

    assert blocks[0].question_start == 10.2
    assert blocks[0].question_end == 20.0
    assert blocks[0].answer_start == 20.7
    assert blocks[0].answer_end == 34.1
    assert blocks[0].response_delay_seconds == 0.7
    assert blocks[0].question == "Представьте, что до запуска пять дней. Как вы составите план?"
    assert blocks[1].type == "condition_change"
    assert blocks[1].question_start == 35.0
    assert blocks[1].answer_start == 42.0


def test_qa_segmentation_normalizes_overlap_and_skips_only_unrecoverable_pair():
    segments = [
        TranscriptSegment(start=0.0, end=2.1, text="Первое задание"),
        TranscriptSegment(start=2.0, end=4.0, text="Первый ответ"),
        TranscriptSegment(start=5.0, end=7.0, text="Условия кейса"),
        TranscriptSegment(start=7.0, end=9.0, text="Уточнение кейса"),
        TranscriptSegment(start=9.0, end=12.0, text="Ответ на кейс"),
        TranscriptSegment(start=13.0, end=14.0, text="Невосстановимый блок"),
        TranscriptSegment(start=15.0, end=17.0, text="Изменение условий"),
        TranscriptSegment(start=17.4, end=20.0, text="Ответ после изменения"),
    ]
    proposed = _ProposedSegmentation(blocks=[
        _ProposedBlock(type="instruction", title="Первый блок", question_segment_ids=[0], answer_segment_ids=[1, 2]),
        _ProposedBlock(type="case", title="Кейс", question_segment_ids=[2, 3], answer_segment_ids=[3, 4]),
        _ProposedBlock(type="question", title="Невозможно", question_segment_ids=[5], answer_segment_ids=[5]),
        _ProposedBlock(type="condition_change", title="Новые условия", question_segment_ids=[6], answer_segment_ids=[7]),
    ])

    blocks = InterviewSegmenter._materialize(proposed, segments)

    assert [block.title for block in blocks] == ["Первый блок", "Кейс", "Новые условия"]
    assert blocks[0].question_end == 2.0
    assert blocks[0].answer_start == 2.0
    assert blocks[0].answer_segment_ids == [1]
    assert blocks[1].question_segment_ids == [2]
    assert blocks[1].answer_segment_ids == [3, 4]
    assert blocks[1].question == "Условия кейса"
    assert blocks[1].answer == "Уточнение кейса Ответ на кейс"
    assert blocks[2].index == 3


def test_rejects_wrong_extension():
    response = client.post("/api/analyze", files={"file": ("sample.txt", b"x", "text/plain")})
    assert response.status_code == 415


def test_video_capture_is_released_when_analysis_fails(monkeypatch):
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, _property):
            return 25

        def release(self):
            self.released = True

    capture = Capture()
    analyzer = VideoAnalyzer()
    monkeypatch.setattr("app.services.video_analysis.cv2.VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        analyzer,
        "_analyze_landmarks",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("frame processing failed")),
    )
    monkeypatch.setattr("app.services.video_analysis.mp", object())

    try:
        analyzer.analyze("input.mp4")
    except RuntimeError:
        pass
    else:
        raise AssertionError("analysis failure was not propagated")

    assert capture.released


def test_report_events_merge_same_type_across_interleaved_events():
    events = [
        TimelineEvent(time=1.0, type="head_movement", description="Движение головы"),
        TimelineEvent(time=1.2, type="facial_change", description="Изменение лица"),
        TimelineEvent(time=1.4, type="head_movement", description="Движение головы"),
        TimelineEvent(time=3.8, type="head_movement", description="Движение головы"),
        TimelineEvent(time=6.4, type="head_movement", description="Движение головы"),
    ]

    merged = _merge_report_events(events)

    assert [(event.time, event.end_time, event.type) for event in merged] == [
        (1.0, 3.8, "head_movement"),
        (1.2, None, "facial_change"),
        (6.4, None, "head_movement"),
    ]
    assert all(event.end_time is None for event in events)


def test_calibration_uses_robust_personal_baseline():
    assert _robust_median([0.29, 0.30, 0.31, 0.30, 0.95]) == 0.3


def test_gaze_is_relative_to_personal_baseline_with_hysteresis():
    assert _classify_gaze(0.58, baseline=0.58) == "center"
    assert _classify_gaze(0.50, baseline=0.58) == "left"
    assert _classify_gaze(0.535, baseline=0.58, previous="left") == "left"
    assert _classify_gaze(0.55, baseline=0.58, previous="left") == "center"


def _eye(confidence: float, ratio: float) -> EyeMeasurement:
    return EyeMeasurement(
        center=np.array([0.4, 0.4]),
        iris_center=np.array([0.4, 0.4]),
        gaze_ratio=ratio,
        ear=0.3,
        confidence=confidence,
        reasons=(),
    )


def test_eye_tracking_uses_two_reliable_agreeing_eyes():
    result = _estimate_gaze(_eye(0.9, 0.48), _eye(0.8, 0.52))

    assert result.label == "available"
    assert result.source == "both_eyes"
    assert result.confidence == 0.8
    assert _classify_gaze(result.ratio, baseline=0.5) == "center"


def test_eye_tracking_uses_one_reliable_eye_when_other_is_poor():
    result = _estimate_gaze(_eye(0.85, 0.30), _eye(0.2, 0.95))

    assert result.label == "available"
    assert result.source == "left_eye"
    assert result.ratio == 0.30
    assert result.confidence == 0.85 * 0.70


def test_eye_tracking_rejects_substantially_disagreeing_eyes():
    result = _estimate_gaze(_eye(0.9, 0.20), _eye(0.9, 0.80))

    assert result.label == "unknown"
    assert result.ratio is None
    assert result.source == "eye_disagreement"


def test_eye_tracking_rejects_two_low_confidence_eyes():
    result = _estimate_gaze(_eye(0.35, 0.30), _eye(0.30, 0.70))

    assert result.label == "unknown"
    assert result.ratio is None
    assert result.source == "unavailable"


def test_debounced_state_ignores_short_fluctuations():
    state = _DebouncedState()
    assert state.update(True, required_frames=2) is None
    assert state.update(False, required_frames=2) is None
    assert state.active is False
    assert state.update(True, required_frames=2) is None
    assert state.update(True, required_frames=2) is True
    assert state.active is True
    assert state.update(False, required_frames=2) is None
    assert state.update(True, required_frames=2) is None
    assert state.active is True


def test_llm_validator_accepts_only_explicit_acoustic_pause_label():
    assert not _analysis_requires_rewrite(_llm_analysis(
        summary="Паузы с низкой энергией составляют около 41% времени речи.",
    ))
    for text in (
        "Низкоэнергетические окна соответствуют паузам или тишине между словами.",
        "Паузы с низкой энергией показывают психологическое состояние.",
        "Паузы с низкой энергией составляют 41%; внутренние паузы в монологе.",
    ):
        assert _analysis_requires_rewrite(_llm_analysis(speech_analysis=text))


def test_llm_returns_acoustic_description_without_retry(monkeypatch):
    analysis = _llm_analysis(summary="Паузы с низкой энергией составляют около 41% времени речи.")
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return type("Response", (), {"output_parsed": analysis})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr("app.services.llm_analysis.OpenAI", FakeOpenAI)
    result = LLMAnalyzer(api_key="test-key").analyze(
        VideoMetrics(analyzer="test"), SpeechMetrics(status="completed", pause_ratio=0.41),
    )
    assert result.status == "completed"
    assert result.analysis == analysis
    assert len(calls) == 1
    prompt = calls[0]["input"][0]["content"]
    assert "не отклонение от нормы" in prompt
    assert "не употребляй слово «пауза»" in prompt

def test_llm_retry_repairs_actual_rejected_fields(monkeypatch):
    rejected = _llm_analysis(
        cross_modal_analysis="Появление улыбок соответствует эмоциональным моментам речи, зафиксированным алгоритмом.",
        interpretation=["Это снижение акустической энергии, а не смысловая или психологическая пауза."],
    )
    assert _analysis_requires_rewrite(rejected)
    calls = []
    corrected = _llm_analysis(cross_modal_analysis="Измеренные изменения геометрии лица сопоставлены с речевыми сегментами.")

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return type("Response", (), {"output_parsed": rejected if len(calls) == 1 else corrected})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr("app.services.llm_analysis.OpenAI", FakeOpenAI)
    result = LLMAnalyzer(api_key="test-key").analyze(
        VideoMetrics(analyzer="test"), SpeechMetrics(status="completed"),
    )
    assert result.status == "completed"
    repair = calls[1]["input"][1]["content"]
    assert rejected.model_dump_json() in repair
    assert '"field": "cross_modal_analysis"' in repair
    assert '"field": "interpretation[0]"' in repair
    assert "не заменяй их синонимами" in repair
