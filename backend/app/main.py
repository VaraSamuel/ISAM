# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, shutil
from typing import Optional

from .pipeline import create_new_run, subclassify_selected

DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Allow comma-separated origins via env, default dev origins
CORS_ORIGIN = os.environ.get(
    "CORS_ORIGIN",
    "http://localhost:5173,http://127.0.0.1:5173",
)
ALLOW_ORIGINS = [o.strip() for o in CORS_ORIGIN.split(",") if o.strip()]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=DATA_DIR), name="artifacts")


class SubclassifyRequest(BaseModel):
    run_id: str
    selected_clusters: list[str]
    k: int = 3

    # keep activity/time settings in subclassify too
    t_start: float = 0.0
    t_end: float = -1.0
    dt: float = 1.0

    activity_method: str = "max_z"
    baseline_frac: float = 0.1
    z_thresh: float = 2.5
    min_above_sec: float = 10.0

    auc_thresh: float = 0.0
    mean_thresh: float = 0.0
    prom_thresh: float = 0.0
    min_peaks: int = 1


def _run_dir(run_id: str) -> str:
    return os.path.join(DATA_DIR, run_id)


@app.post("/runs")
async def runs(
    # Accept BOTH keys:
    # - frontend currently sends "excel_file"
    # - older curl examples often send "file"
    excel_file: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),

    k: int = Form(3),

    t_start: float = Form(0.0),
    t_end: float = Form(-1.0),
    dt: float = Form(1.0),

    activity_method: str = Form("max_z"),
    baseline_frac: float = Form(0.1),

    z_thresh: float = Form(2.5),
    min_above_sec: float = Form(10.0),

    auc_thresh: float = Form(0.0),
    mean_thresh: float = Form(0.0),
    prom_thresh: float = Form(0.0),
    min_peaks: int = Form(1),
):
    up = excel_file or file
    if up is None:
        raise HTTPException(
            status_code=422,
            detail="Missing uploaded file. Send multipart field 'excel_file' (frontend) or 'file' (curl).",
        )

    run_id = str(uuid.uuid4())
    out_dir = _run_dir(run_id)
    os.makedirs(out_dir, exist_ok=True)

    excel_path = os.path.join(out_dir, "input.xlsx")
    with open(excel_path, "wb") as f:
        shutil.copyfileobj(up.file, f)

    return create_new_run(
        excel_path=excel_path,
        out_dir=out_dir,
        run_id=run_id,
        k=k,

        t_start=t_start,
        t_end=t_end,
        dt=dt,

        activity_method=activity_method,
        baseline_frac=baseline_frac,

        z_thresh=z_thresh,
        min_above_sec=min_above_sec,

        auc_thresh=auc_thresh,
        mean_thresh=mean_thresh,
        prom_thresh=prom_thresh,
        min_peaks=min_peaks,
    )


@app.post("/runs/subclassify")
async def subclassify(req: SubclassifyRequest):
    out_dir = _run_dir(req.run_id)

    return subclassify_selected(
        out_dir=out_dir,
        run_id=req.run_id,
        selected_clusters=req.selected_clusters,
        k=req.k,

        t_start=req.t_start,
        t_end=req.t_end,
        dt=req.dt,

        activity_method=req.activity_method,
        baseline_frac=req.baseline_frac,

        z_thresh=req.z_thresh,
        min_above_sec=req.min_above_sec,

        auc_thresh=req.auc_thresh,
        mean_thresh=req.mean_thresh,
        prom_thresh=req.prom_thresh,
        min_peaks=req.min_peaks,
    )
