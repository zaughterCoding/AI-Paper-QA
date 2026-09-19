from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level app, so tests can create isolated instances.
    """
    app = FastAPI(title="AI Paper QA Backend")
    app.include_router(router)
    return app


app = create_app()
