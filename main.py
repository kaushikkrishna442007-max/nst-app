"""
NST FastAPI Backend
Jobs stored in Redis (Upstash) so they survive Render restarts.
"""

import os, uuid, time, shutil, json
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import redis

app = FastAPI(title="Neural Style Transfer API")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "")
r = redis.from_url(REDIS_URL, decode_responses=True)

# ── Directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Mount outputs ─────────────────────────────────────────────────────────────
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ── Redis job helpers ─────────────────────────────────────────────────────────
def save_job(job_id: str, data: dict):
    r.set(f"job:{job_id}", json.dumps(data), ex=86400)  # expire after 24h

def load_job(job_id: str) -> dict | None:
    val = r.get(f"job:{job_id}")
    if not val:
        return None
    return json.loads(val)


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
    job_id = str(uuid.uuid4())

    content_ext = os.path.splitext(content.filename)[1] or ".jpg"
    style_ext   = os.path.splitext(style.filename)[1]   or ".jpg"
    content_path = os.path.join(UPLOAD_DIR, f"{job_id}_content{content_ext}")
    style_path   = os.path.join(UPLOAD_DIR, f"{job_id}_style{style_ext}")
    output_path  = os.path.join(OUTPUT_DIR, f"{job_id}_output.jpg")

    with open(content_path, "wb") as f:
        shutil.copyfileobj(content.file, f)
    with open(style_path, "wb") as f:
        shutil.copyfileobj(style.file, f)

    job = {
        "status":   "queued",
        "progress": 0,
        "step":     0,
        "total":    num_steps,
        "log":      [],
        "output":   None,
        "error":    None,
    }
    save_job(job_id, job)

    background_tasks.add_task(
        run_nst_job,
        job_id, content_path, style_path, output_path,
        image_size, num_steps, style_weight, content_weight, tv_weight, lr,
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = load_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return job


@app.get("/result/{job_id}")
def result(job_id: str):
    job = load_job(job_id)
    if not job or job["status"] != "done":
        return JSONResponse(status_code=404, content={"error": "not ready"})
    return FileResponse(job["output"], media_type="image/jpeg",
                        filename="nst_output.jpg")


# ═══════════════════════════════════════════════════════════════════════════════
# Background job runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_nst_job(job_id, content_path, style_path, output_path,
                image_size, num_steps, style_weight, content_weight, tv_weight, lr):

    job = load_job(job_id)
    job["status"] = "running"
    save_job(job_id, job)

    def progress_cb(step, total, losses):
        j = load_job(job_id)
        j["step"]     = step
        j["total"]    = total
        j["progress"] = round(step / total * 100, 1)
        j["log"].append(
            f"step {step:4d} | style={losses['style']:.4f} "
            f"| content={losses['content']:.4f}"
        )
        save_job(job_id, j)

    try:
        import nst_core
        nst_core.run_job(
            content_path   = content_path,
            style_path     = style_path,
            output_path    = output_path,
            image_size     = image_size,
            num_steps      = num_steps,
            style_weight   = style_weight,
            content_weight = content_weight,
            tv_weight      = tv_weight,
            lr             = lr,
            save_every     = max(1, num_steps // 10),
            progress_cb    = progress_cb,
        )
        job = load_job(job_id)
        job["status"]   = "done"
        job["progress"] = 100
        job["output"]   = output_path
        job["log"].append("✓ Complete")
        save_job(job_id, job)

    except Exception as e:
        job = load_job(job_id)
        job["status"] = "error"
        job["error"]  = str(e)
        job["log"].append(f"✗ {e}")
        save_job(job_id, job)

    finally:
        for p in [content_path, style_path]:
            try: os.remove(p)
            except: pass