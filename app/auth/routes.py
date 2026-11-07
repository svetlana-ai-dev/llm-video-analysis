from pathlib import Path
import secrets

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .security import COOKIE_NAME

router = APIRouter(prefix="/auth", tags=["archive authentication"])
ASSETS = Path(__file__).parent / "static"


class LoginInput(BaseModel):
    username: str = Field(max_length=256)
    password: str = Field(max_length=1024)


def service_for(request):
    return request.app.state.archive_auth


def set_cookie(response, service, token):
    response.set_cookie(COOKIE_NAME, token, httponly=True, secure=service.config.cookie_secure,
                        samesite="strict", path="/", max_age=service.config.lifetime_seconds)


def check_csrf(request, service):
    origin = request.headers.get("origin")
    if origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Недопустимый источник запроса")
    session = service.get_session(request.cookies.get(COOKIE_NAME), touch=False)
    supplied = request.headers.get("x-csrf-token", "")
    if not session or not secrets.compare_digest(session.csrf.encode(), supplied.encode()):
        raise HTTPException(403, "Обновите страницу и повторите вход")
    return session


@router.get("/status")
def auth_status(request: Request, response: Response):
    service = service_for(request)
    response.headers["Cache-Control"] = "no-store"
    if not service.config.enabled:
        return {"enabled": False, "authenticated": False}
    if not service.configured:
        return {"enabled": True, "authenticated": False, "configured": False,
                "message": "Авторизация включена. Сначала настройте логин и пароль администратора."}
    session = service.get_session(request.cookies.get(COOKIE_NAME), touch=False)
    if session is None:
        try:
            token, session = service.create_session()
        except RuntimeError as exc:
            raise HTTPException(503, "Повторите попытку позже") from exc
        set_cookie(response, service, token)
    return {"enabled": True, "configured": True, "authenticated": session.authenticated, "csrf": session.csrf}


@router.post("/login")
def login(data: LoginInput, request: Request, response: Response):
    service = service_for(request)
    response.headers["Cache-Control"] = "no-store"
    if not service.config.enabled:
        raise HTTPException(404, "Авторизация отключена")
    if not service.configured:
        raise HTTPException(503, "Учётные данные администратора не настроены")
    check_csrf(request, service)
    ip = request.client.host if request.client else "unknown"
    if not service.allow_attempt(ip):
        raise HTTPException(429, "Слишком много попыток. Подождите минуту.", headers={"Retry-After": "60"})
    if not service.verify_password(data.username, data.password):
        raise HTTPException(401, "Неверный логин или пароль")
    service.revoke(request.cookies.get(COOKIE_NAME))
    token, session = service.create_session(authenticated=True)
    set_cookie(response, service, token)
    return {"authenticated": True, "csrf": session.csrf}


@router.post("/logout")
def logout(request: Request, response: Response):
    service = service_for(request)
    response.headers["Cache-Control"] = "no-store"
    if service.config.enabled:
        check_csrf(request, service)
        service.revoke(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/", secure=service.config.cookie_secure, httponly=True, samesite="strict")
    return {"authenticated": False}


@router.get("/assets/auth.js", include_in_schema=False)
def auth_script():
    return FileResponse(ASSETS / "auth.js", media_type="text/javascript")


@router.get("/assets/auth.css", include_in_schema=False)
def auth_styles():
    return FileResponse(ASSETS / "auth.css", media_type="text/css")
