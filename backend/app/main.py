# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, uuid, shutil

from .pipeline import create_new_run, subclassify_selected

DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Optional: if you want to explicitly add extra allowed origins via env
# e.g. CORS_ORIGINS="https://your-frontend.com,https://another.com"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()
extra_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

app = FastAPI()

# ✅ Good defaults:
# - allow local dev
# - allow any Railway subdomain
# - plus any extra origins from env
allow_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *extra_origins,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_credentials=False,   # ✅ safer; set True only if you truly use cookies/auth
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static artifacts (downloadable outputs)
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=DATA_DIR), name="artifacts")


class SubclassifyRequest(BaseModel):
    run_id: str
    selected_clusters: list[str]
    k: int = 3


@app.post("/runs")
async def runs(
    excel_file: UploadFile = File(...),
    threshold: float = Form(2.5),
    k: int = Form(3),
):
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(DATA_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    input_path = os.path.join(run_dir, "input.xlsx")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(excel_file.file, f)

    return create_new_run(
        excel_path=input_path,
        out_dir=run_dir,
        run_id=run_id,
        threshold=threshold,
        k=k,
    )


@app.post("/runs/subclassify")
async def runs_subclassify(req: SubclassifyRequest):
    run_dir = os.path.join(DATA_DIR, req.run_id)
    return subclassify_selected(
        out_dir=run_dir,
        run_id=req.run_id,
        selected_clusters=req.selected_clusters,
        k=req.k,
    )
