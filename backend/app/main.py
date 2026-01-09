# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, shutil

from .pipeline import create_new_run, subclassify_selected

DATA_DIR = os.environ.get("DATA_DIR", "./data")
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:5173")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# IMPORTANT: you should already have StaticFiles mounting /artifacts
from fastapi.staticfiles import StaticFiles
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=DATA_DIR), name="artifacts")


class SubclassifyRequest(BaseModel):
    run_id: str
    selected_clusters: list[str]   # <-- NEW: multiple clusters
    k: int = 3                     # optional


@app.post("/runs")
async def runs(excel_file: UploadFile = File(...), threshold: float = Form(2.5), k: int = Form(3)):
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(DATA_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # save input to run folder (pipeline expects input.xlsx)
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
