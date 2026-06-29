# Deploying FORGE

FORGE is two deployable pieces:

| Piece | What it is | Where it runs |
|-------|-----------|---------------|
| **Frontend** | Static React/Vite SPA | **Netlify** (or any static host / CDN) |
| **Backend** | FastAPI + ML pipeline | A container host (Render, Railway, Fly.io, AWS) |

Netlify serves static assets and cannot run a long-lived Python server, so the
two are deployed separately and the frontend is pointed at the backend's URL.

---

## 1. Deploy the backend (FastAPI)

The backend ships with [`Dockerfile.api`](Dockerfile.api). `torch` is **not**
installed by default, so the image stays lean — the API always runs in fast
mode (classical models only), and deep-learning models are skipped gracefully.

**Render / Railway / Fly.io (any "deploy from Dockerfile" host):**

1. Point the host at this repo, Dockerfile = `Dockerfile.api`.
2. Set the start command (already the Docker `CMD`):
   ```
   uvicorn forge.api.app:app --host 0.0.0.0 --port 8000
   ```
3. Set environment variables:
   - `OPENAI_API_KEY` — optional; enables LLM profiling/feature-engineering/reports. Without it, heuristic fallbacks are used.
4. Note the public URL it gives you, e.g. `https://forge-api.onrender.com`.
5. Sanity check: `GET https://<your-backend>/api/v1/health` → `{"status":"ok"}`.

> Experiments and artifacts are stored on the local filesystem and in memory.
> For a demo this is fine; for persistence across restarts, mount a volume or
> add object storage.

---

## 2. Deploy the frontend (Netlify)

Config lives in [`netlify.toml`](netlify.toml) — base dir `frontend`, build
`npm run build`, publish `frontend/dist`, with an SPA fallback redirect.

1. **New site from Git** → pick this repo. Netlify reads `netlify.toml`
   automatically (no manual build settings needed).
2. **Site settings → Environment variables**, add:
   ```
   VITE_API_URL = https://<your-backend-url>     # no trailing slash
   ```
   (See [`frontend/.env.example`](frontend/.env.example).)
3. Deploy. The app calls `${VITE_API_URL}/api/v1/...`.

The backend already sends permissive CORS (`allow_origins=["*"]`), so the
cross-origin calls from your Netlify domain work out of the box. Lock CORS down
to your Netlify domain before any real production use.

---

## 3. Local development

```bash
# Backend
pip install -e ".[dev]"
forge serve api --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Leave `VITE_API_URL` unset locally — the Vite dev proxy forwards `/api` to
`http://localhost:8000`.

## 4. Full stack via Docker (single host)

```bash
docker compose up --build
# API: http://localhost:8000   UI: http://localhost:3000
```
Here nginx proxies `/api` to the API container, so `VITE_API_URL` stays empty.
