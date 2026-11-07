from contextlib import contextmanager
import socket

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import LLMResult, SpeechMetrics, VideoMetrics
from app.services import cloud_download as cloud


MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42" + b"fixture"
WEBM = b"\x1a\x45\xdf\xa3\x87\x42\x82\x84webm" + b"fixture"


class Reply:
    def __init__(self, body=MP4, status=200, headers=None):
        self.body, self.status = body, status
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}

    def getheader(self, key, default=None):
        return self.headers.get(key.lower(), default)

    def read1(self, size):
        chunk, self.body = self.body[:size], self.body[size:]
        return chunk


@pytest.fixture(autouse=True)
def no_real_network_or_ai(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real network/AI/video processing forbidden in cloud tests")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    for target in (
        "app.services.analysis_pipeline.VideoAnalyzer.analyze",
        "app.services.analysis_pipeline.AudioAnalyzer.extract_audio",
        "app.services.analysis_pipeline.AudioAnalyzer.analyze",
        "app.services.analysis_pipeline.LLMAnalyzer.analyze",
        "app.services.audio_analysis.OpenAI",
        "app.services.llm_analysis.OpenAI",
        "app.services.interview_segmentation.OpenAI",
    ):
        monkeypatch.setattr(target, forbidden)


def replies(monkeypatch, *responses):
    seen = []
    queue = iter(responses)

    @contextmanager
    def fake_open(url, timeout):
        cloud._validated_url(url)
        seen.append(url)
        response = next(queue)
        if isinstance(response, Exception):
            raise response
        yield response

    monkeypatch.setattr(cloud, "_open_url", fake_open)
    return seen


@pytest.mark.parametrize("body,mime,suffix", [(MP4,"video/mp4",".mp4"),(WEBM,"video/webm",".webm"),(MP4,"application/octet-stream",".mp4")])
def test_download_content_and_filename(monkeypatch, tmp_path, body, mime, suffix):
    replies(monkeypatch, Reply(body, headers={"Content-Type": mime, "Content-Length": str(len(body))}))
    result = cloud.download_video("https://example.com/recording", tmp_path, 1024)
    assert result.path.read_bytes() == body
    assert result.path.suffix == suffix and result.filename.endswith(suffix)
    assert not (tmp_path / "download.part").exists()


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.mp4", "https://user:pass@example.com/a", "http://example.com:8080/a", "https://example.com/\r\nheader", "http://[fe80::1%25eth0]/a", "http://example.com\\@127.0.0.1/a"])
def test_invalid_urls(url):
    with pytest.raises(cloud.DownloadError):
        cloud._validated_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "0.0.0.0", "224.0.0.1", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:8.8.8.8", "64:ff9b::7f00:1", "64:ff9b:1::1"])
def test_blocks_internal_dns_addresses(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))])
    with pytest.raises(cloud.DownloadError, match="внутренние"):
        cloud._public_addresses("example.com", 443)


def test_mixed_dns_answers_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2,1,6,"",(ip,443)) for ip in ("8.8.8.8", "127.0.0.1")])
    with pytest.raises(cloud.DownloadError):
        cloud._public_addresses("example.com",443)


def test_connection_is_pinned_and_keeps_tls_host(monkeypatch):
    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2,1,6,"",("8.8.8.8",443))])
    monkeypatch.setattr(socket, "create_connection", lambda address, **kw: calls.append(address))

    class Connection:
        def __init__(self, host, port, timeout, context):
            assert host == "example.com" and port == 443
            assert context.check_hostname
        def request(self, method, target, headers):
            assert method == "GET" and target == "/video.mp4?q=1"
            assert headers["Accept-Encoding"] == "identity"
            self._create_connection(("example.com",443), timeout=1)
        def getresponse(self):
            return Reply()
        def close(self):
            calls.append("closed")

    monkeypatch.setattr(cloud.http.client, "HTTPSConnection", Connection)
    with cloud._open_url("https://example.com/video.mp4?q=1",1):
        pass
    assert calls == [("8.8.8.8",443), "closed"]


def test_redirect_to_internal_address_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **kw: [(2,1,6,"",("127.0.0.1" if host == "internal.test" else "8.8.8.8",443))])
    class Connection:
        def __init__(self,*a,**kw): pass
        def request(self,*a,**kw): pass
        def getresponse(self): return Reply(status=302, headers={"Location":"https://internal.test/a.mp4"})
        def close(self): pass
    monkeypatch.setattr(cloud.http.client,"HTTPSConnection", Connection)
    with pytest.raises(cloud.DownloadError, match="внутренние"):
        cloud.download_video("https://example.com/a.mp4",tmp_path,1024)


@pytest.mark.parametrize("response,code", [
    (Reply(status=403),422), (Reply(status=404),422), (Reply(status=500),422),
    (Reply(headers={"Content-Length":"2048"}),413),
    (Reply(b"x"*2048),413), (Reply(headers={"Content-Length":"bad"}),422),
    (Reply(headers={"Content-Length":"-1"}),422),
    (Reply(headers={"Content-Length":"100"}),422),
    (Reply(b"<html>login</html>",headers={"Content-Type":"text/html"}),415),
    (Reply(b"<html>login</html>",headers={"Content-Type":"video/mp4"}),415),
    (Reply(b""),415), (Reply(headers={"Content-Type":"video/webm"}),415),
    (Reply(headers={"Content-Encoding":"gzip"}),415),
    (Reply(headers={"Content-Type":"application/pdf"}),415),
    (TimeoutError(),504), (OSError(),422),
])
def test_download_errors_cleanup(monkeypatch, tmp_path, response, code):
    replies(monkeypatch,response)
    with pytest.raises(cloud.DownloadError) as caught:
        cloud.download_video("https://example.com/video.mp4",tmp_path,1024)
    assert caught.value.status_code == code
    assert list(tmp_path.iterdir()) == []


def test_dns_unavailable(monkeypatch,tmp_path):
    def fail(*a,**kw): raise socket.gaierror()
    monkeypatch.setattr(socket,"getaddrinfo",fail)
    with pytest.raises(cloud.DownloadError,match="определить сервер"):
        cloud.download_video("https://missing.example/video.mp4",tmp_path,1024)


def test_redirects_and_limit(monkeypatch,tmp_path):
    seen = replies(monkeypatch,Reply(status=302,headers={"Location":"/other.mp4"}),Reply())
    assert cloud.download_video("https://example.com/a.mp4",tmp_path,1024).path.read_bytes() == MP4
    assert seen[-1] == "https://example.com/other.mp4"
    replies(monkeypatch, *[Reply(status=302,headers={"Location":"/loop"}) for _ in range(6)])
    with pytest.raises(cloud.DownloadError,match="перенаправления"):
        cloud.download_video("https://example.com/a.mp4",tmp_path,1024)


def test_https_downgrade_rejected(monkeypatch,tmp_path):
    replies(monkeypatch,Reply(status=302,headers={"Location":"http://example.com/a.mp4"}))
    with pytest.raises(cloud.DownloadError,match="HTTPS"):
        cloud.download_video("https://example.com/a.mp4",tmp_path,1024)


@pytest.mark.parametrize("url", ["https://drive.google.com/file/d/abc_123/view?usp=sharing", "https://drive.google.com/open?id=abc_123", "https://drive.google.com/uc?id=abc_123&export=download"])
def test_google_drive_public_file(monkeypatch,tmp_path,url):
    seen = replies(monkeypatch,Reply())
    result=cloud.download_video(url,tmp_path,1024)
    assert result.filename == "drive-abc_123.mp4"
    assert seen == ["https://drive.google.com/uc?id=abc_123&export=download"]


def test_drive_confirmation(monkeypatch,tmp_path):
    html=b'<form action="https://drive.usercontent.google.com/download" method="get"><input type="hidden" name="id" value="abc"><input type="hidden" name="export" value="download"><input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="token"></form>'
    seen=replies(monkeypatch,Reply(html,headers={"Content-Type":"text/html"}),Reply())
    assert cloud.download_video("https://drive.google.com/file/d/abc/view",tmp_path,1024).path.read_bytes() == MP4
    assert seen[-1].startswith("https://drive.usercontent.google.com/download?")


@pytest.mark.parametrize("body", [b"<html>Sign in required</html>", b'<form action="http://127.0.0.1/download"><input type="hidden" name="id" value="abc"></form>'])
def test_private_or_unsafe_drive_page(monkeypatch,tmp_path,body):
    replies(monkeypatch,Reply(body,headers={"Content-Type":"text/html"}))
    with pytest.raises(cloud.DownloadError,match="публичный доступ"):
        cloud.download_video("https://drive.google.com/file/d/abc/view",tmp_path,1024)


def test_download_deadline(monkeypatch,tmp_path):
    replies(monkeypatch,Reply())
    clock=iter([0,1,121])
    monkeypatch.setattr(cloud.time,"monotonic",lambda:next(clock))
    with pytest.raises(cloud.DownloadError,match="Время"):
        cloud.download_video("https://example.com/a.mp4",tmp_path,1024)


def test_cloud_route_shared_pipeline_and_archive(monkeypatch,tmp_path):
    monkeypatch.setenv("RESULTS_DIR",str(tmp_path / "results"))
    replies(monkeypatch,Reply())
    calls=[]
    def video(self,path):
        assert path.read_bytes() == MP4
        calls.append("video")
        return VideoMetrics(analyzer="mock")
    monkeypatch.setattr("app.services.analysis_pipeline.VideoAnalyzer.analyze",video)
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.extract_audio",lambda *a: calls.append("audio"))
    monkeypatch.setattr("app.services.analysis_pipeline.AudioAnalyzer.analyze",lambda *a: SpeechMetrics(status="completed",transcript="fixture"))
    monkeypatch.setattr("app.services.analysis_pipeline.LLMAnalyzer.analyze",lambda *a: LLMResult(status="completed"))
    client=TestClient(app)
    response=client.post("/api/analyze-url",json={"url":"https://example.com/meeting.mp4"})
    assert response.status_code == 200 and response.headers["X-Archive-Status"] == "saved"
    result=response.json()
    assert result["filename"] == "meeting.mp4" and result["speech"]["transcript"] == "fixture"
    from pathlib import Path
    saved=Path(result["saved_result_path"])
    assert (saved / "source.mp4").read_bytes() == MP4
    assert (saved / "result.json").is_file()
    assert calls == ["video","audio"]
    # Both entry points use exactly the same service calls and result schema.
    local=client.post("/api/analyze",files={"file":("local.mp4",MP4,"video/mp4")})
    assert local.status_code == 200
    assert set(local.json()) == set(result)
    assert calls == ["video","audio","video","audio"]


def test_cloud_route_failure_does_not_analyze_and_local_still_available(monkeypatch):
    replies(monkeypatch,Reply(status=404))
    client=TestClient(app)
    response=client.post("/api/analyze-url",json={"url":"https://example.com/missing.mp4"})
    assert response.status_code == 422 and "недоступно" in response.json()["detail"]
    assert client.post("/api/analyze",files={"file":("wrong.txt",b"x")}).status_code == 415
    assert client.post("/api/analyze-url",json={"url":""}).status_code == 422


def test_cloud_ui_is_additive():
    from pathlib import Path
    page=Path("app/templates/index.html").read_text(encoding="utf-8")
    script=Path("app/static/app.js").read_text(encoding="utf-8")
    assert 'id="cloud-form"' in page and "Ссылка на видеозапись" in page
    assert 'id="file"' in page and 'id="drop"' in page
    assert 'fetch("/api/analyze-url"' in script and 'fetch("/api/analyze"' in script
