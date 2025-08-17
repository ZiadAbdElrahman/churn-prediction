from fastapi import FastAPI
from .dependencies import get_settings
from .engine.controller.churn_detection import router as churn_router


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(title="Thmanyah Churn API", version="1.0")

    api_prefix = cfg["server"]["prefix"].rstrip("/")
    app.include_router(churn_router, prefix=api_prefix, tags=["churn"])

    @app.get("/")
    async def root():
        return {"message": "Welcome to Churn Detection Service"}

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
