"""Optional module. Install once; AUTH_ENABLED=false restores open archive access."""


def install_auth(app, config=None):
    from .config import AuthConfig
    from .security import AuthService
    from .middleware import ArchiveAuthMiddleware
    from .routes import router
    service = AuthService(config or AuthConfig.from_env())
    app.state.archive_auth = service
    app.include_router(router)
    app.add_middleware(ArchiveAuthMiddleware, service=service)
    return service
