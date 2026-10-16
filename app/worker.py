"""Background photo processor.

Runs inside the web service as one thread. It only wakes when photos arrive
(or once at startup), so it never polls the database while idle — Neon can sleep.
"""
import json
import threading
import traceback

from . import faces, storage
from .db import pool
from .matching import match_photo, vec

_wake = threading.Event()
_stop = threading.Event()
_thread: threading.Thread | None = None


def wake():
    _wake.set()


def _claim(cur):
    cur.execute(
        """UPDATE photos SET status = 'processing'
           WHERE id = (SELECT id FROM photos WHERE status = 'pending' ORDER BY id
                       FOR UPDATE SKIP LOCKED LIMIT 1)
           RETURNING id, tenant_id, event_id, r2_key"""
    )
    return cur.fetchone()


def process_one() -> bool:
    """Process a single pending photo. Returns False when the queue is empty."""
    with pool.connection() as conn, conn.cursor() as cur:
        photo = _claim(cur)
    if not photo:
        return False
    try:
        data = storage.get_bytes(photo["r2_key"])
        found, quality = faces.photo_faces(data)
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM faces WHERE photo_id = %s", (photo["id"],))   # idempotent re-runs
            for f in found:
                cur.execute(
                    "INSERT INTO faces (photo_id, event_id, bbox, embedding) VALUES (%s, %s, %s, %s::vector)",
                    (photo["id"], photo["event_id"], json.dumps(f["bbox"]), vec(f["embedding"])),
                )
            matched = match_photo(cur, photo["id"], photo["tenant_id"])
            cur.execute(
                "UPDATE photos SET status = 'processed', face_count = %s, quality = %s, error = NULL WHERE id = %s",
                (len(found), quality, photo["id"]),
            )
        print(f"[worker] photo {photo['id']}: {len(found)} faces, {matched} matched", flush=True)
    except faces.FaceError as e:
        _fail(photo["id"], "rejected", str(e))
    except Exception as e:  # keep the worker alive no matter what one photo does
        traceback.print_exc()
        _fail(photo["id"], "failed", f"{type(e).__name__}: {e}"[:500])
    return True


def _fail(photo_id: int, status: str, msg: str):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE photos SET status = %s, error = %s WHERE id = %s", (status, msg, photo_id))


def purge_expired_face_data():
    """Delete face fingerprints (and their matches) for events past their retention date."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """DELETE FROM faces f USING events e
               WHERE f.event_id = e.id AND e.face_data_expires_at IS NOT NULL AND e.face_data_expires_at <= now()"""
        )
        if cur.rowcount:
            print(f"[worker] purged {cur.rowcount} expired face records", flush=True)


def _loop():
    # Startup sweep: anything a crash left half-done goes back in the queue
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE photos SET status = 'pending' WHERE status = 'processing'")
        purge_expired_face_data()
    except Exception:
        traceback.print_exc()
    _wake.set()
    while not _stop.is_set():
        _wake.wait()
        _wake.clear()
        if _stop.is_set():
            break
        try:
            while not _stop.is_set() and process_one():
                pass
            purge_expired_face_data()
        except Exception:
            traceback.print_exc()


def start():
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_loop, name="photo-worker", daemon=True)
        _thread.start()


def stop():
    global _thread
    _stop.set()
    _wake.set()
    if _thread is not None:
        _thread.join(timeout=30)   # let a photo in progress finish before the DB pool closes
        _thread = None
    _stop.clear()
