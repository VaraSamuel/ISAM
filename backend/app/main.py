# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, uuid, shutil

from .pipeline import create_new_run, subclassify_selected

DATA_DIR = os.environ.get("DATA_DIR", "./data")

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()
extra_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

app = FastAPI()

allow_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *extra_origins,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=DATA_DIR), name="artifacts")


class SubclassifyRequest(BaseModel):
    run_id: str
    selected_clusters: list[str]
    k: int = 3

    # time slicing
    t_start: float = 0.0
    t_end: float = -1.0
    dt: float = 1.0

    # activity detection config
    activity_method: str = "max_z"          # max_z | auc | mean_over_baseline | prominence | composite
    z_thresh: float = 2.5                  # for max_z / composite
    auc_thresh: float = 0.0                # for auc / composite
    mean_thresh: float = 0.0               # for mean_over_baseline / composite
    prom_thresh: float = 0.0               # for prominence / composite
    baseline_frac: float = 0.1             # fraction of window used for baseline
    min_peaks: int = 1                     # for prominence (>=1 means at least one peak)


@app.post("/runs")
async def runs(
    excel_file: UploadFile = File(...),

    # clustering
    k: int = Form(3),

    # time slicing
    t_start: float = Form(0.0),
    t_end: float = Form(-1.0),
    dt: float = Form(1.0),

    # activity detection config
    activity_method: str = Form("max_z"),
    z_thresh: float = Form(2.5),
    auc_thresh: float = Form(0.0),
    mean_thresh: float = Form(0.0),
    prom_thresh: float = Form(0.0),
    baseline_frac: float = Form(0.1),
    min_peaks: int = Form(1),
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
        k=k,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        activity_method=activity_method,
        z_thresh=z_thresh,
        auc_thresh=auc_thresh,
        mean_thresh=mean_thresh,
        prom_thresh=prom_thresh,
        baseline_frac=baseline_frac,
        min_peaks=min_peaks,
    )


@app.post("/runs/subclassify")
async def runs_subclassify(req: SubclassifyRequest):
    run_dir = os.path.join(DATA_DIR, req.run_id)
    return subclassify_selected(
        out_dir=run_dir,
        run_id=req.run_id,
        selected_clusters=req.selected_clusters,
        k=req.k,
        t_start=req.t_start,
        t_end=req.t_end,
        dt=req.dt,
        activity_method=req.activity_method,
        z_thresh=req.z_thresh,
        auc_thresh=req.auc_thresh,
        mean_thresh=req.mean_thresh,
        prom_thresh=req.prom_thresh,
        baseline_frac=req.baseline_frac,
        min_peaks=req.min_peaks,
    )
