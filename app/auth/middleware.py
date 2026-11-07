from starlette.responses import JSONResponse

from .security import COOKIE_NAME


class ArchiveAuthMiddleware:
    def __init__(self, app, service):
        self.app = app
        self.service = service

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        protected = path == "/api/archive" or path.startswith("/api/archive/")
        auth_request = path in ("/auth/status", "/auth/login", "/auth/logout")
        if scope["type"] != "http" or not (auth_request or (protected and self.service.config.enabled)):
            return await self.app(scope, receive, send)
        from starlette.requests import Request
        request = Request(scope)
        session = self.service.get_session(request.cookies.get(COOKIE_NAME), touch=protected)
        if protected and self.service.config.enabled and (not session or not session.authenticated):
            response = JSONResponse({"detail": "Для просмотра архива войдите в систему."}, status_code=401,
                                    headers={"Cache-Control": "no-store"})
            return await response(scope, receive, send)

        async def no_cache(message):
            if message["type"] == "http.response.start":
                headers = [(key, value) for key, value in message.get("headers", []) if key.lower() != b"cache-control"]
                message = {**message, "headers": headers + [(b"cache-control", b"no-store")]}
            await send(message)

        await self.app(scope, receive, no_cache)
