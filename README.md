# Neural Style Transfer — Full Stack

```
Frontend (index.html)  ──POST /stylize──►  FastAPI (main.py)
                        ◄──poll /status──   runs nst_core.py
                        ◄──GET /result ──   returns JPEG
```

## Files

| File | Role |
|---|---|
| `main.py` | FastAPI server — routing, job queue, file I/O |
| `nst_core.py` | Your original NST code (zero logic changes) |
| `index.html` | Frontend — drop here or host on Netlify/GitHub Pages |
| `requirements.txt` | Python dependencies |
| `render.yaml` | One-click Render deploy config |

---

## Local development (test before deploying)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Start the API
uvicorn main:app --reload --port 8000

# 3. Open index.html in your browser
# The const API = "http://localhost:8000" line at the top of the
# <script> block already points there — no changes needed locally.
```

Visit http://localhost:8000/docs to see the auto-generated API docs.

---

## Deploy backend to Render (free tier)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Python version:** 3.11
5. Click Deploy
6. Copy your URL: `https://your-app-name.onrender.com`

---

## Deploy backend to Railway (alternative)

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

---

## Connect frontend to your deployed backend

Open `index.html` and find this line near the top of the `<script>` block:

```js
const API = "http://localhost:8000";   // ← replace after deploy
```

Replace with your Render/Railway URL:

```js
const API = "https://your-app-name.onrender.com";
```

---

## Deploy frontend

**Option A — Netlify Drop (easiest)**
1. Go to https://netlify.com/drop
2. Drag `index.html` into the browser
3. Done — instant public URL

**Option B — GitHub Pages**
1. Put `index.html` in a repo
2. Settings → Pages → Deploy from branch (main / root)

**Option C — Serve from FastAPI itself**
Add to `main.py`:
```python
from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return open("index.html").read()
```
Then deploy `index.html` alongside `main.py` and set `API = ""` (empty string).

---

## GPU note

- **Free Render/Railway:** CPU only. 512px × 300 steps ≈ 8–15 minutes.
- **Render paid (Standard+):** Can request GPU instances.
- **Google Colab:** Still the best free GPU option — use the original notebook for fast runs, the API for the web interface.

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `POST /stylize` | multipart/form-data | Submit job, returns `{job_id}` |
| `GET /status/{job_id}` | — | Poll progress: `{status, progress, step, total, log}` |
| `GET /result/{job_id}` | — | Download output JPEG when `status=="done"` |
| `GET /health` | — | Health check |
