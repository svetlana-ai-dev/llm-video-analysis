"""Bounded public-URL downloads with DNS pinning (no environment proxies)."""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit, unquote


MAX_REDIRECTS = 5
DOWNLOAD_SECONDS = 120
SOCKET_TIMEOUT = 15
HTML_LIMIT = 256 * 1024
DRIVE_HOSTS = {"drive.google.com", "drive.usercontent.google.com"}
NAT64_NETWORKS = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


class DownloadError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DownloadedVideo:
    path: Path
    filename: str


def _validated_url(url: str):
    if not url or len(url) > 4096 or any(ord(c) < 33 or ord(c) == 127 for c in url) or "\\" in url:
        raise DownloadError("Укажите корректную HTTP/HTTPS-ссылку на видео.")
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or parts.password is not None:
            raise ValueError
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if port != (443 if parts.scheme == "https" else 80):
            raise ValueError
        host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if "%" in host:
            raise ValueError
    except (ValueError, UnicodeError):
        raise DownloadError("Разрешены HTTP/HTTPS-ссылки без логина и пароля, со стандартным портом.") from None
    return parts, host, port


def _public_addresses(host: str, port: int) -> list[str]:
    try:
        addresses = list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    except OSError:
        raise DownloadError("Адрес недоступен: не удалось определить сервер.", 422) from None
    if not addresses:
        raise DownloadError("Адрес недоступен: сервер не найден.", 422)
    for address in addresses:
        ip = ipaddress.ip_address(address)
        # Reject mapped/transition IPv6 as well as private, loopback, link-local,
        # reserved and multicast addresses, including mixed public/private DNS.
        if not ip.is_global or ip.is_multicast or (ip.version == 6 and (
            ip.ipv4_mapped or ip.sixtofour or ip.teredo or any(ip in network for network in NAT64_NETWORKS)
        )):
            raise DownloadError("Ссылки на локальные и внутренние сетевые адреса запрещены.")
    return addresses


@contextmanager
def _open_url(url: str, timeout: float):
    parts, host, port = _validated_url(url)
    addresses = _public_addresses(host, port)
    connection = (http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
                  if parts.scheme == "https" else http.client.HTTPConnection(host, port, timeout=timeout))
    # Connect to the vetted numeric address, while preserving the original host
    # for Host, TLS SNI and certificate verification. No second DNS lookup.
    connection._create_connection = lambda _address, timeout, source_address=None: socket.create_connection(
        (addresses[0], port), timeout=timeout, source_address=source_address
    )
    try:
        target = urlunsplit(("", "", parts.path or "/", parts.query, ""))
        connection.request("GET", target, headers={"Accept-Encoding": "identity", "User-Agent": "VideoAnalysis/1.0"})
        response = connection.getresponse()
        yield response
    finally:
        connection.close()


def _drive_source(url: str) -> tuple[str, str | None]:
    parts, host, _ = _validated_url(url)
    if host != "drive.google.com":
        return url, None
    query = parse_qs(parts.query)
    match = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)(?:/view|/preview|/edit)?/?", parts.path)
    file_id = match.group(1) if match else (query.get("id", [None])[0] if parts.path in {"/open", "/uc"} else None)
    if not file_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", file_id):
        raise DownloadError("Укажите публичную ссылку Google Drive на отдельный файл.")
    params = {"id": file_id, "export": "download"}
    resourcekey = query.get("resourcekey", [None])[0]
    if resourcekey:
        params["resourcekey"] = resourcekey
    return "https://drive.google.com/uc?" + urlencode(params), file_id


class _DriveForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current = (attrs.get("action", ""), attrs.get("method", "get").lower(), {})
        elif tag == "input" and self.current is not None and attrs.get("type", "").lower() == "hidden":
            name = attrs.get("name")
            if name in {"id", "export", "confirm", "uuid", "resourcekey"}:
                self.current[2][name] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form" and self.current is not None:
            self.forms.append(self.current)
            self.current = None


def _drive_confirmation(html: bytes, url: str, file_id: str) -> str:
    parser = _DriveForm()
    parser.feed(html.decode("utf-8", errors="replace"))
    for action, method, params in parser.forms:
        target = urljoin(url, action)
        parts, host, _ = _validated_url(target)
        if (method == "get" and parts.scheme == "https" and host in DRIVE_HOSTS
                and parts.path in {"/uc", "/download"} and params.get("id") == file_id
                and params.get("confirm") and params.get("export") == "download"):
            return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))
    raise DownloadError("Google Drive не предоставил видео. Проверьте публичный доступ и разрешение скачивания.", 422)


def _media_suffix(prefix: bytes) -> str:
    if len(prefix) >= 16 and prefix[4:8] == b"ftyp":
        size = int.from_bytes(prefix[:4], "big")
        if 16 <= size <= len(prefix):
            brands = [prefix[8:12]] + [prefix[i:i + 4] for i in range(16, size, 4)]
            if any(brand in {b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"dash", b"M4V "} for brand in brands):
                return ".mp4"
    # EBML DocType (0x4282) must explicitly identify WebM, not generic Matroska.
    if prefix.startswith(b"\x1a\x45\xdf\xa3") and b"\x42\x82\x84webm" in prefix[:4096]:
        return ".webm"
    raise DownloadError("Ссылка не содержит MP4/WebM-видео: получен другой формат или повреждённый файл.", 415)


def download_video(url: str, directory: Path, max_bytes: int) -> DownloadedVideo:
    url, drive_id = _drive_source(url.strip())
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    confirmed = False
    path = directory / "download.part"
    try:
        for hop in range(MAX_REDIRECTS + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DownloadError("Время скачивания истекло. Попробуйте другую ссылку.", 504)
            with _open_url(url, min(SOCKET_TIMEOUT, remaining)) as response:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location or hop == MAX_REDIRECTS:
                        raise DownloadError("Не удалось скачать видео: слишком много или некорректные перенаправления.", 422)
                    target = urljoin(url, location)
                    parts, _, _ = _validated_url(target)
                    if urlsplit(url).scheme == "https" and parts.scheme != "https":
                        raise DownloadError("Небезопасное перенаправление с HTTPS на HTTP.")
                    url = target
                    continue
                if response.status != 200:
                    raise DownloadError(f"Видео недоступно (HTTP {response.status}). Проверьте ссылку и разрешение скачивания.", 422)
                encoding = response.getheader("Content-Encoding", "identity").lower()
                if encoding not in {"", "identity"}:
                    raise DownloadError("Сервер вернул неподдерживаемое сжатие файла.", 415)
                content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
                length = response.getheader("Content-Length")
                try:
                    declared_size = int(length) if length is not None else None
                    if declared_size is not None and declared_size < 0:
                        raise ValueError
                except ValueError:
                    raise DownloadError("Сервер сообщил некорректный размер файла.", 422) from None
                if declared_size is not None and declared_size > max_bytes:
                    raise DownloadError("Видео превышает допустимый размер загрузки.", 413)
                html = content_type in {"text/html", "application/xhtml+xml"}
                if html and not (drive_id and not confirmed and urlsplit(url).hostname in DRIVE_HOSTS):
                    raise DownloadError("Получена веб-страница вместо видео. Нужна прямая ссылка или публичный файл Google Drive.", 415)
                if not html and content_type not in {"", "video/mp4", "video/webm", "application/octet-stream", "binary/octet-stream"}:
                    raise DownloadError("Сервер вернул неподдерживаемый тип содержимого. Нужен MP4 или WebM.", 415)
                size, prefix = 0, bytearray()
                limit = min(max_bytes, HTML_LIMIT) if html else max_bytes
                with path.open("wb") as target:
                    while True:
                        if time.monotonic() >= deadline:
                            raise DownloadError("Время скачивания истекло. Попробуйте другую ссылку.", 504)
                        # read1 returns after one buffered/socket read, allowing
                        # the wall-clock deadline to stop slow trickle streams.
                        chunk = response.read1(min(64 * 1024, limit - size + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > limit:
                            raise DownloadError("Видео или ответ сервера превышает допустимый размер загрузки.", 413)
                        if len(prefix) < 65536:
                            prefix.extend(chunk[:65536 - len(prefix)])
                        target.write(chunk)
                if declared_size is not None and size != declared_size:
                    raise DownloadError("Видео скачано не полностью. Повторите попытку.", 422)
                if html:
                    url = _drive_confirmation(path.read_bytes(), url, drive_id)
                    confirmed = True
                    continue
                suffix = _media_suffix(bytes(prefix))
                if content_type.startswith("video/") and content_type != {".mp4": "video/mp4", ".webm": "video/webm"}[suffix]:
                    raise DownloadError("Тип содержимого не соответствует формату видео.", 415)
                media_path = directory / ("input" + suffix)
                path.rename(media_path)
                name = Path(unquote(urlsplit(url).path)).name
                if drive_id:
                    name = f"drive-{drive_id}{suffix}"
                else:
                    name = re.sub(r'[^\w. -]', '_', name)[:150]
                    if Path(name).suffix.lower() != suffix:
                        name = (Path(name).stem or "cloud-video") + suffix
                return DownloadedVideo(media_path, name)
        raise DownloadError("Слишком много перенаправлений при скачивании видео.", 422)
    except DownloadError:
        raise
    except (TimeoutError, socket.timeout):
        raise DownloadError("Сервер не ответил вовремя. Повторите попытку или загрузите файл с компьютера.", 504) from None
    except (OSError, http.client.HTTPException, ValueError):
        raise DownloadError("Не удалось безопасно скачать видео. Проверьте ссылку или загрузите файл с компьютера.", 422) from None
    finally:
        path.unlink(missing_ok=True)
