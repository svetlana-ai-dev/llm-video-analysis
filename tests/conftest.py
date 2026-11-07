import pytest


@pytest.fixture(autouse=True)
def block_real_meeting_summary_api(monkeypatch):
    """Existing and new pipeline tests must never spend API credits."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Real meeting-summary OpenAI calls are forbidden in tests")
    monkeypatch.setattr("app.services.meeting_summary.OpenAI", forbidden)
