import json
from pathlib import Path
from dataclasses import replace
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import AnalysisResponse, VideoMetrics, SpeechMetrics, LLMResult
from app.services.result_storage import save_analysis_result, list_archive


client = TestClient(app)


@pytest.fixture(autouse=True)
def archive_without_auth(monkeypatch):
    # Stage-one archive tests explicitly exercise the backwards-compatible mode.
    service = app.state.archive_auth
    monkeypatch.setattr(service, "config", replace(service.config, enabled=False))


def sample_result():
    return AnalysisResponse(
        filename="interview.mp4",
        video=VideoMetrics(analyzer="test", blink_count=7, events=[{
            "time": 1.2, "end_time": 1.4, "type": "blink", "description": "Моргание"
        }]),
        speech=SpeechMetrics(status="completed", transcript="Вопрос. Ответ.",
            segments=[{"start": 0, "end": 1, "text": "Вопрос."}],
            qa_segmentation={"status": "completed", "blocks": [{
                "index": 1, "type": "question", "title": "Вопрос", "question": "Вопрос.",
                "question_start": 0, "question_end": 1, "answer": "Ответ.",
                "answer_start": 1, "answer_end": 2, "response_delay_seconds": 0,
            }]}),
        llm=LLMResult(status="completed", analysis={
            "summary": "Резюме", "facial_analysis": "Мимика", "gaze_analysis": "Взгляд",
            "head_movement_analysis": "Голова", "speech_analysis": "Речь",
            "cross_modal_analysis": "Сопоставление", "timeline_insights": ["Временной вывод"],
            "observed_facts": ["Факт"], "interpretation": ["Вывод"], "limitations": ["Ограничение"],
        }), processing_seconds=3.5,
    )


def forbid_processing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Archive must never process video or call external APIs")
    for target in (
        "app.services.analysis_pipeline.VideoAnalyzer.analyze", "app.services.analysis_pipeline.AudioAnalyzer.extract_audio",
        "app.services.analysis_pipeline.AudioAnalyzer.analyze", "app.services.analysis_pipeline.LLMAnalyzer.analyze",
        "app.services.interview_segmentation.InterviewSegmenter.segment",
        "app.services.audio_analysis.OpenAI", "app.services.llm_analysis.OpenAI",
        "app.services.interview_segmentation.OpenAI",
    ):
        monkeypatch.setattr(target, forbidden)


def test_archive_roundtrip_full_result_video_and_no_processing(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    source = tmp_path / "input.mp4"
    source.write_bytes(b"original-video-bytes")
    result = sample_result()
    directory = save_analysis_result(result, source)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    forbid_processing(monkeypatch)
    records = client.get("/api/archive").json()
    assert len(records) == 1
    record = records[0]
    detail = client.get(f"/api/archive/{record['id']}").json()
    assert detail["result"] == result.model_dump(mode="json")
    assert detail["result"] == json.loads(before["result.json"])
    assert client.get(record["video_url"]).content == source.read_bytes()
    ranged = client.get(record["video_url"], headers={"Range": "bytes=0-7"})
    assert ranged.status_code == 206
    assert ranged.content == source.read_bytes()[:8]
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_legacy_archive_is_read_only_and_video_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    directory = tmp_path / "old_result"
    directory.mkdir()
    payload = sample_result().model_dump(mode="json")
    (directory / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    before = (directory / "result.json").read_bytes()
    forbid_processing(monkeypatch)
    record = client.get("/api/archive").json()[0]
    assert record["id"].startswith("legacy-")
    assert record["warning"] and not record["video_available"]
    assert client.get(f"/api/archive/{record['id']}").json()["result"] == payload
    assert client.get(f"/api/archive/{record['id']}/video").status_code == 404
    assert (directory / "result.json").read_bytes() == before
    assert list(directory.iterdir()) == [directory / "result.json"]
    assert client.get("/api/archive/unknown").status_code == 404


def test_same_filename_keeps_distinct_video_result_pairs(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    source = tmp_path / "input.mp4"
    first = sample_result()
    source.write_bytes(b"first-video")
    first_directory = save_analysis_result(first, source)
    before = {p.name: p.read_bytes() for p in first_directory.iterdir()}
    second = sample_result()
    second.video.blink_count = 19
    source.write_bytes(b"second-video")
    second_directory = save_analysis_result(second, source)
    assert first_directory != second_directory
    records = client.get("/api/archive").json()
    assert len({r["id"] for r in records}) == 2
    for record in records:
        restored = client.get(f"/api/archive/{record['id']}").json()["result"]
        expected = b"first-video" if restored["video"]["blink_count"] == 7 else b"second-video"
        assert client.get(record["video_url"]).content == expected
    assert {p.name: p.read_bytes() for p in first_directory.iterdir()} == before


def test_storage_failure_preserves_analysis_response(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    expected = sample_result()
    monkeypatch.setattr("app.services.analysis_pipeline.VideoAnalyzer.analyze", lambda *a: expected.video)
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.extract_audio", lambda *a: None)
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.analyze", lambda *a: expected.speech)
    monkeypatch.setattr("app.services.analysis_pipeline.LLMAnalyzer.analyze", lambda *a: expected.llm)
    def fail_copy(*args):
        raise OSError("disk full")
    monkeypatch.setattr("app.services.result_storage.shutil.copyfile", fail_copy)
    response = client.post("/api/analyze", files={"file": ("sample.mp4", b"video", "video/mp4")})
    assert response.status_code == 200
    assert response.headers["X-Archive-Status"] == "failed"
    data = response.json()
    assert data["saved_result_path"] is None
    assert data["video"] == expected.video.model_dump(mode="json")
    assert data["speech"] == expected.speech.model_dump(mode="json")
    assert data["llm"] == expected.llm.model_dump(mode="json")
    assert list_archive() == []
    assert not list((tmp_path / "results").iterdir())


def test_incomplete_and_unsafe_archives_are_not_exposed(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    directory = save_analysis_result(sample_result(), source)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_file"] = "../input.mp4"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    pending = tmp_path / ".pending-test"
    pending.mkdir()
    (pending / "result.json").write_bytes((directory / "result.json").read_bytes())
    records = list_archive()
    assert len(records) == 1
    assert not records[0]["video_available"]
