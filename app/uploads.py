import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import storage, worker
from .auth import current_account_id
from .db import pool

router = APIRouter(prefix="/api/memories", tags=["uploads"])

ALLOWED_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
MAX_PHOTO_BYTES = 30 * 1024 * 1024
MAX_FILES_PER_BATCH = 50


def require_staff(account_id: int = Depends(current_account_id)) -> dict:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, tenant_id, role FROM accounts WHERE id = %s", (account_id,))
        acc = cur.fetchone()
    if not acc or acc["role"] not in ("photographer", "admin"):
        raise HTTPException(403, "This page is for Lux photographers and staff.")
    return acc


@router.get("/staff/setup")
def staff_setup(staff: dict = Depends(require_staff)):
    """Events and stations for the upload page's dropdowns."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, slug, name, starts_at FROM events
               WHERE tenant_id = %s AND (ends_at IS NULL OR ends_at > now() - interval '14 days')
               ORDER BY starts_at NULLS LAST, id""",
            (staff["tenant_id"],),
        )
        events = cur.fetchall()
        cur.execute("SELECT id, code, kind, name FROM stations WHERE tenant_id = %s ORDER BY code", (staff["tenant_id"],))
        stations = cur.fetchall()
    return {"events": events, "stations": stations, "max_bytes": MAX_PHOTO_BYTES, "max_batch": MAX_FILES_PER_BATCH}


class FileIn(BaseModel):
    name: str = Field(max_length=255)
    type: str
    size: int


class PresignIn(BaseModel):
    event_id: int
    station_id: int
    files: list[FileIn]


@router.post("/photos/presign")
def presign(body: PresignIn, staff: dict = Depends(require_staff)):
    if not body.files or len(body.files) > MAX_FILES_PER_BATCH:
        raise HTTPException(400, f"Send between 1 and {MAX_FILES_PER_BATCH} files at a time.")
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, slug FROM events WHERE id = %s AND tenant_id = %s", (body.event_id, staff["tenant_id"]))
        event = cur.fetchone()
        cur.execute("SELECT id, code, kind FROM stations WHERE id = %s AND tenant_id = %s",
                    (body.station_id, staff["tenant_id"]))
        station = cur.fetchone()
        if not event or not station:
            raise HTTPException(400, "Unknown event or station.")
        cur.execute("SELECT slug FROM tenants WHERE id = %s", (staff["tenant_id"],))
        tenant_slug = cur.fetchone()["slug"]
        pool_name = "station" if station["kind"] == "photographer" else "event"

        out, skipped = [], []
        for f in body.files:
            ext = ALLOWED_TYPES.get(f.type)
            if not ext:
                skipped.append({"name": f.name, "reason": "Only JPEG, PNG or WebP photos."})
                continue
            if f.size <= 0 or f.size > MAX_PHOTO_BYTES:
                skipped.append({"name": f.name, "reason": "File is empty or larger than 30 MB."})
                continue
            station_code = re.sub(r"[^A-Za-z0-9_-]", "", station["code"]) or "x"
            key = f"{tenant_slug}/{event['slug']}/{station_code}/{uuid.uuid4().hex}.{ext}"
            cur.execute(
                """INSERT INTO photos (tenant_id, event_id, station_id, pool, r2_key, status, uploaded_by, original_name, bytes)
                   VALUES (%s, %s, %s, %s, %s, 'uploading', %s, %s, %s) RETURNING id""",
                (staff["tenant_id"], event["id"], station["id"], pool_name, key, staff["id"], f.name[:255], f.size),
            )
            out.append({"photo_id": cur.fetchone()["id"], "name": f.name, "url": storage.presign_put(key, f.type),
                        "content_type": f.type})
    return {"uploads": out, "skipped": skipped}


class CompleteIn(BaseModel):
    photo_ids: list[int]


@router.post("/photos/complete")
def complete(body: CompleteIn, staff: dict = Depends(require_staff)):
    if len(body.photo_ids) > MAX_FILES_PER_BATCH:
        raise HTTPException(400, "Too many photos in one call.")
    queued, missing = 0, []
    with pool.connection() as conn, conn.cursor() as cur:
        for pid in body.photo_ids:
            cur.execute(
                "SELECT id, r2_key FROM photos WHERE id = %s AND uploaded_by = %s AND status = 'uploading'",
                (pid, staff["id"]),
            )
            p = cur.fetchone()
            if not p:
                continue
            obj = storage.head(p["r2_key"])
            if not obj:
                missing.append(pid)
                continue
            cur.execute("UPDATE photos SET status = 'pending', bytes = %s WHERE id = %s",
                        (obj.get("ContentLength"), pid))
            queued += 1
    if queued:
        worker.wake()
    return {"queued": queued, "missing": missing}


@router.get("/photos/stats")
def stats(event_id: int, staff: dict = Depends(require_staff)):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT status, count(*) AS n, coalesce(sum(face_count), 0) AS faces FROM photos
               WHERE event_id = %s AND tenant_id = %s GROUP BY status""",
            (event_id, staff["tenant_id"]),
        )
        rows = cur.fetchall()
        cur.execute(
            """SELECT count(DISTINCT m.account_id) AS people, count(*) AS matches FROM matches m
               JOIN photos p ON p.id = m.photo_id WHERE p.event_id = %s AND p.tenant_id = %s""",
            (event_id, staff["tenant_id"]),
        )
        m = cur.fetchone()
    return {"by_status": {r["status"]: r["n"] for r in rows},
            "faces": sum(r["faces"] for r in rows), "people_matched": m["people"], "matches": m["matches"]}
