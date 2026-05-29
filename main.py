"""
NST FastAPI Backend
Wraps the existing neural_style_transfer_colab.py without modifying it.
All NST logic lives in nst_core.py (identical to original, just importable).
"""

import os, uuid, time, asyncio, shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Neural Style Transfer API")

# ── CORS (allow your frontend origin) ────────────────────────────────────────
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Job store (in-memory; fine for single-worker deploy) ─────────────────────
JOBS: dict[str, dict] = {}

UPLOAD_DIR  = "uploads"
OUTPUT_DIR  = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Mount outputs so /outputs/<file> is directly downloadable ─────────────────
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "NST API running"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/stylize")
async def stylize(
    background_tasks: BackgroundTasks,
    content: UploadFile = File(...),
    style:   UploadFile = File(...),
    image_size:     int   = Form(512),
    num_steps:      int   = Form(300),
    style_weight:   float = Form(1_000_000),
    content_weight: float = Form(1),
    tv_weight:      float = Form(1),
    lr:             float = Form(0.02),
):
    """
    Accepts two images + hyperparameters.
    Returns a job_id immediately; poll /status/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())

    # Save uploads
    content_path = os.path.join(UPLOAD_DIR, f"{job_id}_content{os.path.splitext(content.filename)[1]}")
    style_path   = os.path.join(UPLOAD_DIR, f"{job_id}_style{os.path.splitext(style.filename)[1]}")
    output_path  = os.path.join(OUTPUT_DIR, f"{job_id}_output.jpg")

    with open(content_path, "wb") as f:
        shutil.copyfileobj(content.file, f)
    with open(style_path, "wb") as f:
        shutil.copyfileobj(style.file, f)

    JOBS[job_id] = {
        "status":   "queued",
        "progress": 0,
        "step":     0,
        "total":    num_steps,
        "log":      [],
        "output":   None,
        "error":    None,
    }

    # Run NST in background thread (it's CPU/GPU-bound, not async)
    background_tasks.add_task(
        run_nst_job,
        job_id, content_path, style_path, output_path,
        image_size, num_steps, style_weight, content_weight, tv_weight, lr,
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return job


@app.get("/result/{job_id}")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse(status_code=404, content={"error": "not ready"})
    return FileResponse(job["output"], media_type="image/jpeg",
                        filename="nst_output.jpg")


# ═══════════════════════════════════════════════════════════════════════════════
# Background job runner — calls your original NST code unchanged
# ═══════════════════════════════════════════════════════════════════════════════

def run_nst_job(job_id, content_path, style_path, output_path,
                image_size, num_steps, style_weight, content_weight, tv_weight, lr):
    """
    Runs the NST pipeline from nst_core.py (your original code, untouched).
    Updates JOBS[job_id] with live progress so /status can be polled.
    """
    job = JOBS[job_id]
    job["status"] = "running"

    def progress_cb(step, total, losses):
        """Called by nst_core every SAVE_EVERY steps."""
        job["step"]     = step
        job["total"]    = total
        job["progress"] = round(step / total * 100, 1)
        job["log"].append(
            f"step {step:4d} | style={losses['style']:.4f} "
            f"| content={losses['content']:.4f}"
        )

    try:
        import nst_core
        nst_core.run_job(
            content_path    = content_path,
            style_path      = style_path,
            output_path     = output_path,
            image_size      = image_size,
            num_steps       = num_steps,
            style_weight    = style_weight,
            content_weight  = content_weight,
            tv_weight       = tv_weight,
            lr              = lr,
            save_every      = max(1, num_steps // 10),
            progress_cb     = progress_cb,
        )
        job["status"]   = "done"
        job["progress"] = 100
        job["output"]   = output_path
        job["log"].append("✓ Complete")

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        job["log"].append(f"✗ {e}")

    finally:
        # Clean up uploads
        for p in [content_path, style_path]:
            try: os.remove(p)
            except: pass
