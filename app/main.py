from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import faces, storage, worker
from .auth import current_account_id, router as auth_router
from .db import pool
from .enroll import router as enroll_router
from .uploads import router as uploads_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    faces.load_models()   # downloads ~39 MB of models on first boot, then cached
    if storage.enabled():
        worker.start()
    else:
        print("[startup] R2 not configured — photo uploads and processing are off.", flush=True)
    yield
    worker.stop()
    pool.close()


app = FastAPI(title="Lux Memories", lifespan=lifespan, docs_url=None, redoc_url=None)
app.include_router(auth_router)
app.include_router(enroll_router)
app.include_router(uploads_router)


@app.get("/api/memories/health")
def health():
    return {"ok": True, "service": "lux-memories", "uploads": storage.enabled()}


@app.get("/api/memories/me")
def me(account_id: int = Depends(current_account_id)):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT a.email, a.name, a.role, a.created_at,
                      a.consent_security_at IS NOT NULL AS consent_security,
                      a.consent_matching_at IS NOT NULL AS consent_matching,
                      a.selfie_enrolled_at IS NOT NULL AS selfie_enrolled,
                      (SELECT count(DISTINCT photo_id) FROM matches m WHERE m.account_id = a.id) AS photos_found
               FROM accounts a WHERE a.id = %s""",
            (account_id,),
        )
        return cur.fetchone() or {}
