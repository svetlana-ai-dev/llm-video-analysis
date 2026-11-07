"""Configuration of the optional archive authentication module."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool = False
    username: str = ""
    password_hash: str = ""
    cookie_secure: bool = False
    idle_seconds: int = 1800
    lifetime_seconds: int = 28800

    @classmethod
    def from_env(cls):
        return cls(
            # Only an explicit false disables protection; a typo must not expose data.
            enabled=os.getenv("AUTH_ENABLED", "false").strip().lower() != "false",
            username=os.getenv("AUTH_ADMIN_USERNAME", ""),
            password_hash=os.getenv("AUTH_ADMIN_PASSWORD_HASH", ""),
            cookie_secure=os.getenv("AUTH_COOKIE_SECURE", "false").strip().lower() != "false",
        )
