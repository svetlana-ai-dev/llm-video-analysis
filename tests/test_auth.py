import json

import pytest
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.auth import install_auth
from app.auth.config import AuthConfig
from app.auth.security import COOKIE_NAME
from app.services.result_storage import save_analysis_result
from tests.test_archive import sample_result, forbid_processing


@pytest.fixture(scope="module")
def password_hash():
    return PasswordHasher().hash("Demo password 2026!")


def make_client(password_hash, enabled=True, **settings):
    application = FastAPI()
    application.include_router(router)
    service = install_auth(application, AuthConfig(enabled=enabled, username="admin",
                           password_hash=password_hash, **settings))
    return TestClient(application), service


def login(client, password="Demo password 2026!"):
    state = client.get("/auth/status").json()
    return client.post("/auth/login", json={"username": "admin", "password": password},
                       headers={"Origin": "http://testserver", "X-CSRF-Token": state["csrf"]})


@pytest.mark.parametrize("enabled", [True, False])
def test_archive_full_result_video_ranges_and_no_processing(enabled, password_hash, monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    result = sample_result()
    directory = save_analysis_result(result, source)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    record_id = json.loads(before["metadata.json"])["id"]
    client, _ = make_client(password_hash, enabled)
    forbid_processing(monkeypatch)
    if enabled:
        for endpoint in ("/api/archive", f"/api/archive/{record_id}", f"/api/archive/{record_id}/video", "/api/archive/unknown"):
            for method in ("GET", "HEAD"):
                denied = client.request(method, endpoint, headers={"Range": "bytes=0-2"})
                assert denied.status_code == 401
                assert "original-video" not in denied.text
        # Encoded and trailing slash paths cannot bypass the boundary.
        assert client.get("/api/%61rchive").status_code == 401
        assert client.get("/api/archive/").status_code == 401
        assert login(client).status_code == 200
    else:
        assert client.get("/auth/status").json()["enabled"] is False
    assert len(client.get("/api/archive").json()) == 1
    detail = client.get(f"/api/archive/{record_id}")
    assert detail.json()["result"] == result.model_dump(mode="json")
    assert client.get(f"/api/archive/{record_id}/video").content == source.read_bytes()
    ranged = client.get(f"/api/archive/{record_id}/video", headers={"Range": "bytes=0-2"})
    assert ranged.status_code == 206 and ranged.content == b"ori"
    if enabled:
        assert detail.headers["Cache-Control"] == "no-store"
        assert ranged.headers["Cache-Control"] == "no-store"
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_wrong_password_cookie_rotation_logout_and_forgery(password_hash):
    client, service = make_client(password_hash)
    assert login(client, "wrong").status_code == 401
    anonymous_token = client.cookies.get(COOKIE_NAME)
    response = login(client)
    assert response.status_code == 200
    token = client.cookies.get(COOKIE_NAME)
    assert token != anonymous_token
    assert service.get_session(anonymous_token) is None
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/auth/status").json()["authenticated"] is True
    csrf = response.json()["csrf"]
    response = client.post("/auth/logout", headers={"Origin": "http://testserver", "X-CSRF-Token": csrf})
    assert response.status_code == 200
    assert service.get_session(token) is None
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/api/archive").status_code == 401
    client.cookies.set(COOKIE_NAME, "forged")
    assert client.get("/api/archive").status_code == 401


def test_csrf_origin_and_rate_limit(password_hash):
    client, _ = make_client(password_hash)
    state = client.get("/auth/status").json()
    data = {"username": "admin", "password": "Demo password 2026!"}
    assert client.post("/auth/login", json=data).status_code == 403
    assert client.post("/auth/login", json=data, headers={"Origin": "https://evil.example", "X-CSRF-Token": state["csrf"]}).status_code == 403
    assert client.post("/auth/login", json=data, headers={"Origin": "http://testserver", "X-CSRF-Token": "forged"}).status_code == 403
    for _ in range(5):
        assert login(client, "wrong").status_code == 401
    assert login(client).status_code == 429


def test_expiry_and_status_poll_does_not_extend_idle(password_hash, monkeypatch):
    import app.auth.security as security
    clock = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: clock[0])
    client, service = make_client(password_hash, idle_seconds=10, lifetime_seconds=20)
    assert login(client).status_code == 200
    clock[0] = 109
    assert client.get("/auth/status").json()["authenticated"]
    clock[0] = 111
    assert client.get("/api/archive").status_code == 401
    assert login(client).status_code == 200
    clock[0] = 119
    assert client.get("/api/archive").status_code == 200
    clock[0] = 127
    assert client.get("/api/archive").status_code == 200
    clock[0] = 132
    assert client.get("/api/archive").status_code == 401


def test_unconfigured_enabled_stays_closed_and_disabled_stays_open(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    for invalid_hash in ("", "not-a-hash"):
        client, _ = make_client(invalid_hash)
        state = client.get("/auth/status").json()
        assert state["configured"] is False
        assert client.get("/api/archive").status_code == 401
        assert client.post("/auth/login", json={"username": "admin", "password": "test"}).status_code == 503
    client, _ = make_client("", enabled=False)
    assert client.get("/api/archive").status_code == 200


def test_secure_cookie_and_analysis_route_is_unaffected(password_hash):
    client, _ = make_client(password_hash, cookie_secure=True)
    cookie = client.get("/auth/status").headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie
    # The analyzer's existing validation runs, not the archive authentication gate.
    assert client.post("/api/analyze", files={"file": ("bad.txt", b"text", "text/plain")}).status_code == 415


@pytest.mark.parametrize("enabled", [True, False])
def test_environment_switch_uses_same_installation(enabled, password_hash, monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("AUTH_ENABLED", str(enabled).lower())
    monkeypatch.setenv("AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("AUTH_ADMIN_PASSWORD_HASH", password_hash)
    application = FastAPI()
    application.include_router(router)
    service = install_auth(application)
    client = TestClient(application)
    assert service.config.enabled is enabled
    assert client.get("/api/archive").status_code == (401 if enabled else 200)
    if enabled:
        assert login(client).status_code == 200
        assert client.get("/api/archive").status_code == 200


def test_invalid_enable_flag_does_not_open_archive(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "treu")
    monkeypatch.setenv("AUTH_ADMIN_PASSWORD_HASH", "")
    application = FastAPI()
    application.include_router(router)
    install_auth(application)
    assert TestClient(application).get("/api/archive").status_code == 401


def test_setup_stores_hash_and_preserves_existing_settings(monkeypatch, tmp_path):
    from app.auth import setup
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "app" / "auth" / "setup.py"))
    monkeypatch.setattr("builtins.input", lambda prompt: "demo")
    monkeypatch.setattr(setup, "_terminal_available", lambda: True)
    monkeypatch.setattr(setup, "getpass", lambda prompt: "Demo password 2026!")
    env_path = tmp_path / ".env"
    env_path.write_text("PORT=3000\nOPENAI_API_KEY=fixture-only\nAUTH_ENABLED=false\n", encoding="utf-8")
    setup.main()
    content = env_path.read_text(encoding="utf-8")
    assert "Demo password 2026!" not in content
    assert "PORT=3000" in content and "OPENAI_API_KEY=fixture-only" in content
    assert content.count("AUTH_ENABLED=") == 1 and "AUTH_ENABLED=true" in content
    from dotenv import dotenv_values
    values = dotenv_values(env_path)
    assert PasswordHasher().verify(values["AUTH_ADMIN_PASSWORD_HASH"], "Demo password 2026!")


@pytest.mark.parametrize("kind", ["english", "russian", "mixed"])
def test_setup_unicode_hidden_roundtrip(kind, monkeypatch, tmp_path, capsys):
    from app.auth import setup
    from dotenv import dotenv_values
    from secrets import randbelow
    alphabet = {"english": range(65, 91), "russian": range(1040, 1072), "mixed": [65, 90, 1040, 1071, 1105]}[kind]
    password = "".join(chr(alphabet[randbelow(len(alphabet))]) for _ in range(16))
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "app" / "auth" / "setup.py"))
    monkeypatch.setattr("builtins.input", lambda prompt: "demo")
    monkeypatch.setattr(setup, "_terminal_available", lambda: True)
    reads = iter([password, password])
    monkeypatch.setattr(setup, "getpass", lambda prompt: next(reads))
    setup.main()
    env = tmp_path / ".env"
    values = dotenv_values(env)
    verified = PasswordHasher().verify(values["AUTH_ADMIN_PASSWORD_HASH"], password)
    hidden = password not in capsys.readouterr().out and password not in env.read_text(encoding="utf-8")
    assert verified and hidden


@pytest.mark.parametrize("case,expected", [("empty", "пустой"), ("short", "короткий"), ("long", "длинный"), ("mismatch", "не совпадают")])
def test_setup_separate_validation_no_env_write(case, expected, monkeypatch, tmp_path, capsys):
    from app.auth import setup
    from secrets import token_urlsafe
    password = token_urlsafe(24)
    confirmation = password
    if case == "empty":
        password = ""
    elif case == "short":
        password = password[:11]
    elif case == "long":
        password = password * 40
    else:
        confirmation = token_urlsafe(24)
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "app" / "auth" / "setup.py"))
    monkeypatch.setattr("builtins.input", lambda prompt: "demo")
    monkeypatch.setattr(setup, "_terminal_available", lambda: True)
    values = iter([password, confirmation])
    monkeypatch.setattr(setup, "getpass", lambda prompt: next(values))
    env = tmp_path / ".env"
    env.write_text("PORT=3000\n", encoding="utf-8")
    before = env.read_bytes()
    with pytest.raises(SystemExit) as error:
        setup.main()
    assert expected in str(error.value)
    assert env.read_bytes() == before
    output = capsys.readouterr().out + str(error.value)
    concealed = not password or password not in output
    assert concealed


@pytest.mark.parametrize("failure", ["warning", "eof", "oserror", "redirect"])
def test_setup_never_falls_back_to_echo(failure, monkeypatch):
    from app.auth import setup
    import warnings
    from getpass import GetPassWarning
    monkeypatch.setattr(setup, "_terminal_available", lambda: failure != "redirect")
    def failed_read(prompt):
        if failure == "warning":
            warnings.warn("synthetic unsupported terminal", GetPassWarning)
            pytest.fail("Echo fallback must not be reached")
        if failure == "eof":
            raise EOFError
        if failure == "oserror":
            raise OSError
        pytest.fail("Redirected input must not be read")
    monkeypatch.setattr(setup, "getpass", failed_read)
    with pytest.raises(SystemExit) as error:
        setup._hidden_input("synthetic prompt")
    assert "--gui" in str(error.value)


def test_setup_gui_masks_both_password_fields(monkeypatch, tmp_path, capsys):
    from app.auth import setup
    import sys
    from types import SimpleNamespace
    from secrets import token_urlsafe
    password = token_urlsafe(24)
    calls = []
    destroyed = []
    replies = iter(["demo", password, password])
    def ask(*args, **kwargs):
        calls.append(kwargs.get("show"))
        return next(replies)
    fake_root = SimpleNamespace(withdraw=lambda: None, destroy=lambda: destroyed.append(True))
    dialog = SimpleNamespace(askstring=ask)
    monkeypatch.setitem(sys.modules, "tkinter", SimpleNamespace(Tk=lambda: fake_root, simpledialog=dialog))
    monkeypatch.setitem(sys.modules, "tkinter.simpledialog", dialog)
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "app" / "auth" / "setup.py"))
    setup.main(["--gui"])
    assert calls == [None, "*", "*"] and destroyed == [True]
    hidden = password not in capsys.readouterr().out and password not in (tmp_path / ".env").read_text(encoding="utf-8")
    assert hidden


def test_setup_windows_getpass_unicode_and_backspace(monkeypatch):
    # Test the actual Python Windows reader with synthetic Unicode console events.
    import getpass as library
    import sys
    from types import SimpleNamespace
    from secrets import randbelow
    reader = getattr(library, "win_getpass", None)
    if reader is None:
        pytest.skip("Windows-only console reader")
    alphabet = [65, 90, 1040, 1071, 1105]
    password = "".join(chr(alphabet[randbelow(len(alphabet))]) for _ in range(16))
    keys = iter(password + chr(88) + chr(8) + chr(13))
    console_output = []
    monkeypatch.setattr(library, "msvcrt", SimpleNamespace(getwch=lambda: next(keys), putwch=console_output.append))
    monkeypatch.setattr(sys, "stdin", sys.__stdin__)
    read = reader("synthetic prompt")
    correct = read == password and password not in "".join(console_output)
    assert correct
