"""Single administrator and revocable in-memory sessions, independent of analysis."""
from collections import deque
from dataclasses import dataclass
import hashlib
import secrets
from threading import RLock
import time

from .config import AuthConfig

COOKIE_NAME = "video_archive_session"


@dataclass
class Session:
    csrf: str
    authenticated: bool
    created: float
    accessed: float


class AuthService:
    def __init__(self, config: AuthConfig):
        self.config = config
        self.sessions = {}
        self.attempts = {}
        self.global_attempts = deque()
        self.lock = RLock()
        self.hasher = None
        self.configured = False
        if config.enabled and config.username and config.password_hash:
            try:
                from argon2 import PasswordHasher, extract_parameters, Type
                self.configured = extract_parameters(config.password_hash).type == Type.ID
                self.hasher = PasswordHasher()
            except (ImportError, ValueError):
                pass

    @staticmethod
    def key(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def _prune(self, now):
        for key, session in list(self.sessions.items()):
            lifetime = self.config.lifetime_seconds if session.authenticated else 600
            if now - session.created >= lifetime or now - session.accessed >= self.config.idle_seconds:
                del self.sessions[key]
        for ip, attempts in list(self.attempts.items()):
            while attempts and now - attempts[0] >= 60:
                attempts.popleft()
            if not attempts:
                del self.attempts[ip]
        while self.global_attempts and now - self.global_attempts[0] >= 60:
            self.global_attempts.popleft()

    def get_session(self, token, touch=True):
        if not token or len(token) > 256:
            return None
        with self.lock:
            now = time.monotonic()
            self._prune(now)
            session = self.sessions.get(self.key(token))
            if session and touch:
                session.accessed = now
            return session

    def create_session(self, authenticated=False):
        with self.lock:
            now = time.monotonic()
            self._prune(now)
            # Bound pre-login state, including anonymous status requests.
            if len(self.sessions) >= 1024:
                oldest = next((key for key, value in self.sessions.items() if not value.authenticated), None)
                if oldest:
                    del self.sessions[oldest]
                else:
                    raise RuntimeError("Session capacity reached")
            token = secrets.token_urlsafe(32)
            session = Session(secrets.token_urlsafe(32), authenticated, now, now)
            self.sessions[self.key(token)] = session
            return token, session

    def revoke(self, token):
        if token:
            with self.lock:
                self.sessions.pop(self.key(token), None)

    def allow_attempt(self, ip):
        with self.lock:
            now = time.monotonic()
            self._prune(now)
            attempts = self.attempts.setdefault(ip, deque())
            if len(attempts) >= 5 or len(self.global_attempts) >= 20:
                return False
            attempts.append(now)
            self.global_attempts.append(now)
            return True

    def verify_password(self, username, password):
        if not self.configured or len(username) > 256 or len(password) > 1024:
            return False
        from argon2.exceptions import VerificationError, InvalidHashError
        try:
            valid_password = self.hasher.verify(self.config.password_hash, password)
            return valid_password and secrets.compare_digest(username.encode(), self.config.username.encode())
        except (VerificationError, InvalidHashError):
            return False
