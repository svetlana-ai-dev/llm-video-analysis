import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.schemas import MeetingSummaryContent, MeetingSummaryResult, SpeechMetrics
from app.services.meeting_summary import MeetingSummaryAnalyzer
from app.services.result_storage import save_analysis_result
from tests.test_archive import sample_result, forbid_processing


TRANSCRIPT = "Решили выпустить обновление. Анна подготовит отчёт до пятницы. Вопрос бюджета остаётся открытым."
SENTENCES = ["Обсуждалось обновление.", "Участники решили его выпустить.", "Анна подготовит отчёт.", "Назван срок до пятницы.", "Бюджет остаётся открытым вопросом."]


def content(**overrides):
    return MeetingSummaryContent(**{
        "summary_sentences": SENTENCES,
        "decisions": ["Решили выпустить обновление."],
        "tasks": [{"task": "подготовит отчёт", "assignee": "Анна", "deadline": "до пятницы", "source_quote": "Анна подготовит отчёт до пятницы."}],
        "open_questions": ["Вопрос бюджета остаётся открытым."], **overrides,
    })


def stub_api(monkeypatch, parsed=None, error=None):
    calls = []
    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            if error:
                raise error
            return type("Reply", (), {"output_parsed": parsed})()
    class Client:
        def __init__(self, **kwargs): self.responses = Responses()
    monkeypatch.setattr("app.services.meeting_summary.OpenAI", Client)
    return calls


def test_summary_from_transcript_only(monkeypatch):
    calls = stub_api(monkeypatch, content())
    result = MeetingSummaryAnalyzer(api_key="fixture").generate(SpeechMetrics(status="completed", transcript=TRANSCRIPT))
    assert result.status == "completed"
    assert len(result.content.summary_sentences) == 5
    assert result.content.tasks[0].assignee == "Анна"
    assert calls[0]["input"][1]["content"] == TRANSCRIPT
    assert calls[0]["text_format"] is MeetingSummaryContent


def test_no_invented_decisions_or_tasks_for_plain_conversation(monkeypatch):
    parsed = content(summary_sentences=["Собеседники поздоровались.", "Обсудили погоду.", "Конкретных поручений не прозвучало.", "Решений участники не принимали.", "Нерешённых вопросов не обозначено."], decisions=[], tasks=[], open_questions=[])
    stub_api(monkeypatch, parsed)
    result = MeetingSummaryAnalyzer(api_key="fixture").generate(SpeechMetrics(status="completed", transcript="Привет! Сегодня солнечно. Да, хорошая погода."))
    assert result.status == "completed"
    assert result.content.decisions == result.content.tasks == result.content.open_questions == []


@pytest.mark.parametrize("overrides", [
    {"decisions": ["Нанять нового сотрудника."]},
    {"open_questions": ["Кто оплатит поездку?"]},
    {"tasks": [{"task":"подготовит отчёт", "assignee":"Борис", "deadline":None, "source_quote":"Анна подготовит отчёт до пятницы."}]},
    {"tasks": [{"task":"подготовит отчёт", "assignee":"Анна", "deadline":"завтра", "source_quote":"Анна подготовит отчёт до пятницы."}]},
    {"tasks": [{"task":"удалить базу", "source_quote":"Анна подготовит отчёт до пятницы."}]},
    {"tasks": [{"task":"проверить договор", "source_quote":"Борис проверит договор завтра."}]},
])
def test_unsupported_extractions_are_rejected(monkeypatch, overrides):
    stub_api(monkeypatch, content(**overrides))
    result = MeetingSummaryAnalyzer(api_key="fixture").generate(SpeechMetrics(status="completed", transcript=TRANSCRIPT))
    assert result.status == "failed" and result.content is None


def test_missing_assignee_and_deadline_remain_null(monkeypatch):
    stub_api(monkeypatch, content(decisions=[], open_questions=[], tasks=[{"task":"Подготовить отчёт", "source_quote":"Подготовить отчёт."}]))
    result = MeetingSummaryAnalyzer(api_key="fixture").generate(SpeechMetrics(status="completed", transcript="Подготовить отчёт."))
    assert result.status == "completed"
    assert result.content.tasks[0].assignee is None and result.content.tasks[0].deadline is None


@pytest.mark.parametrize("status,text,key", [("failed", TRANSCRIPT,"fixture"),("unavailable",TRANSCRIPT,"fixture"),("completed","  ","fixture"),("completed",TRANSCRIPT,"")])
def test_unavailable_does_not_call_api(monkeypatch,status,text,key):
    calls=stub_api(monkeypatch,content())
    assert MeetingSummaryAnalyzer(api_key=key).generate(SpeechMetrics(status=status,transcript=text)).status == "unavailable"
    assert calls == []


@pytest.mark.parametrize("error", [TimeoutError("secret transcript"),RuntimeError("secret key")])
def test_api_errors_safe_and_separate(monkeypatch,error):
    stub_api(monkeypatch,error=error)
    result=MeetingSummaryAnalyzer(api_key="fixture").generate(SpeechMetrics(status="completed",transcript=TRANSCRIPT))
    assert result.status == "failed" and result.content is None
    assert "secret" not in result.message


def test_missing_structured_response(monkeypatch):
    stub_api(monkeypatch)
    assert MeetingSummaryAnalyzer(api_key="fixture").generate(SpeechMetrics(status="completed",transcript=TRANSCRIPT)).status == "failed"


def test_five_to_eight_sentences_required():
    for count in (0,4,9):
        with pytest.raises(ValidationError):
            content(summary_sentences=["Предложение."] * count)


def test_save_json_markdown_and_archive_read_without_api(monkeypatch,tmp_path):
    monkeypatch.setenv("RESULTS_DIR",str(tmp_path))
    monkeypatch.setattr(app.state.archive_auth,"config",replace(app.state.archive_auth.config,enabled=False))
    result=sample_result()
    result.speech.transcript=TRANSCRIPT
    result.meeting_summary=MeetingSummaryResult(status="completed",content=content())
    directory=save_analysis_result(result)
    stored=json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert stored["meeting_summary"] == result.meeting_summary.model_dump(mode="json")
    markdown=(directory / "summary.md").read_text(encoding="utf-8")
    assert markdown.index(TRANSCRIPT) < markdown.index("## Краткое содержание встречи") < markdown.index("## Вопросы и ответы")
    assert "Ответственный: Анна" in markdown and "Срок: до пятницы" in markdown
    assert all(sentence in markdown for sentence in SENTENCES)
    forbid_processing(monkeypatch)
    monkeypatch.setattr("app.services.analysis_pipeline.MeetingSummaryAnalyzer.generate",lambda *a:pytest.fail("Archive must not summarize"))
    client=TestClient(app)
    record=client.get("/api/archive").json()[0]
    assert client.get(f"/api/archive/{record['id']}").json()["result"] == stored


def test_empty_lists_render_explicitly_in_markdown(monkeypatch,tmp_path):
    monkeypatch.setenv("RESULTS_DIR",str(tmp_path))
    result=sample_result()
    result.meeting_summary=MeetingSummaryResult(status="completed",content=content(decisions=[],tasks=[],open_questions=[]))
    markdown=(save_analysis_result(result) / "summary.md").read_text(encoding="utf-8")
    assert markdown.count("Не выявлены") == 3


def test_old_archive_without_field_is_unchanged(monkeypatch,tmp_path):
    monkeypatch.setenv("RESULTS_DIR",str(tmp_path))
    monkeypatch.setattr(app.state.archive_auth,"config",replace(app.state.archive_auth.config,enabled=False))
    data=sample_result().model_dump(mode="json")
    data.pop("meeting_summary")
    directory=tmp_path / "legacy"
    directory.mkdir()
    path=directory / "result.json"
    path.write_text(json.dumps(data),encoding="utf-8")
    before=path.read_bytes()
    forbid_processing(monkeypatch)
    monkeypatch.setattr("app.services.analysis_pipeline.MeetingSummaryAnalyzer.generate",lambda *a:pytest.fail("Legacy archive must not summarize"))
    client=TestClient(app)
    record=client.get("/api/archive").json()[0]
    assert client.get(f"/api/archive/{record['id']}").json()["result"] == data
    assert path.read_bytes() == before


@pytest.mark.parametrize("behavior_failed,summary_raises",[(True,False),(False,True)])
def test_pipeline_independence_and_preservation(monkeypatch,tmp_path,behavior_failed,summary_raises):
    from app.models.schemas import LLMResult
    monkeypatch.setenv("RESULTS_DIR",str(tmp_path / "results"))
    sample=sample_result()
    monkeypatch.setattr("app.services.analysis_pipeline.VideoAnalyzer.analyze",lambda *a:sample.video)
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.extract_audio",lambda *a:None)
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.analyze",lambda *a:SpeechMetrics(status="completed",transcript=TRANSCRIPT))
    monkeypatch.setattr("app.services.analysis_pipeline.LLMAnalyzer.analyze",lambda *a:LLMResult(status="failed" if behavior_failed else "completed"))
    def generate(*a):
        if summary_raises: raise RuntimeError("fixture failure")
        return MeetingSummaryResult(status="completed",content=content())
    monkeypatch.setattr("app.services.analysis_pipeline.MeetingSummaryAnalyzer.generate",generate)
    response=TestClient(app).post("/api/analyze",files={"file":("fixture.mp4",b"synthetic","video/mp4")})
    assert response.status_code == 200 and response.headers["X-Archive-Status"] == "saved"
    data=response.json()
    assert data["meeting_summary"]["status"] == ("failed" if summary_raises else "completed")
    assert data["llm"]["status"] == ("failed" if behavior_failed else "completed")
    assert (Path(data["saved_result_path"]) / "result.json").is_file()


def test_ui_section_order():
    page=Path("app/templates/index.html").read_text(encoding="utf-8")
    assert page.index('data-role="transcript"') < page.index("Краткое содержание встречи") < page.index("Вопросы и ответы по времени")
