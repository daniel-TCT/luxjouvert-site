import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import faces
from .auth import current_account_id
from .db import pool
from .matching import match_account

router = APIRouter(prefix="/api/memories", tags=["enrollment"])

MAX_SELFIE_BYTES = 8 * 1024 * 1024
ENROLL_PER_HOUR = 10
_attempts: dict[int, deque] = defaultdict(deque)   # per-account rate limit (single instance)


def _rate_limit(account_id: int):
    q, now = _attempts[account_id], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= ENROLL_PER_HOUR:
        raise HTTPException(429, "Too many selfie attempts. Try again in an hour.")
    q.append(now)


def _vec(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class Consent(BaseModel):
    name: str | None = None
    security: bool          # selfie used to confirm identity at entry
    matching: bool          # selfie used to find this person in event photos


@router.post("/consent")
def set_consent(body: Consent, account_id: int = Depends(current_account_id)):
    name = (body.name or "").strip()[:120] or None
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE accounts SET
                 name = COALESCE(%s, name),
                 consent_security_at = CASE WHEN %s THEN COALESCE(consent_security_at, now()) END,
                 consent_matching_at = CASE WHEN %s THEN COALESCE(consent_matching_at, now()) END
               WHERE id = %s""",
            (name, body.security, body.matching, account_id),
        )
        if not body.matching:
            # Withdrawing photo-matching consent removes existing matches
            cur.execute("DELETE FROM matches WHERE account_id = %s", (account_id,))
        else:
            match_account(cur, account_id)   # no-op until a selfie is enrolled
        if not body.security and not body.matching:
            cur.execute(
                "UPDATE accounts SET selfie_embedding = NULL, selfie_enrolled_at = NULL WHERE id = %s",
                (account_id,),
            )
    return {"ok": True}


@router.post("/enroll")
async def enroll(selfie: UploadFile = File(...), account_id: int = Depends(current_account_id)):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT consent_security_at, consent_matching_at FROM accounts WHERE id = %s",
                    (account_id,))
        row = cur.fetchone()
    if not row or (row["consent_security_at"] is None and row["consent_matching_at"] is None):
        raise HTTPException(403, "Please accept the consent terms before taking your selfie.")

    _rate_limit(account_id)
    data = await selfie.read(MAX_SELFIE_BYTES + 1)
    if len(data) > MAX_SELFIE_BYTES:
        raise HTTPException(413, "That photo is too large.")

    try:
        emb = faces.selfie_embedding(data)
    except faces.FaceError as e:
        raise HTTPException(422, str(e))
    finally:
        del data   # the selfie image itself is never stored — only the fingerprint

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE accounts SET selfie_embedding = %s::vector, selfie_enrolled_at = now()
               WHERE id = %s""",
            (_vec(emb), account_id),
        )
        # New selfie → old matches may be stale; rematch against photos already on file
        cur.execute("DELETE FROM matches WHERE account_id = %s", (account_id,))
        found = match_account(cur, account_id)
    return {"ok": True, "enrolled": True, "photos_found": found}


@router.delete("/face")
def delete_face_data(account_id: int = Depends(current_account_id)):
    """'Delete my face data': removes fingerprint, matches and both consents."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM matches WHERE account_id = %s", (account_id,))
        cur.execute(
            """UPDATE accounts SET selfie_embedding = NULL, selfie_enrolled_at = NULL,
                 consent_security_at = NULL, consent_matching_at = NULL
               WHERE id = %s""",
            (account_id,),
        )
    return {"ok": True}
