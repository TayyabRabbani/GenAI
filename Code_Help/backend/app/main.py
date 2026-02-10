from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from backend.app.routes.review import router

app = FastAPI(title="AI Code Reviewer")

app.include_router(router)

BASE_DIR = Path(__file__).resolve().parents[2]

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "web_ui.html")
