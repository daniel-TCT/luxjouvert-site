"""Cloudflare R2 (S3-compatible) access. Photos go browser → R2 directly via presigned URLs."""
import os

import boto3
from botocore.config import Config

R2_ENDPOINT = os.getenv("R2_ENDPOINT")            # https://<account_id>.r2.cloudflarestorage.com
R2_BUCKET = os.getenv("R2_BUCKET", "lux-memories")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

_client = None


def enabled() -> bool:
    return bool(R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY)


def client():
    global _client
    if _client is None:
        if not enabled():
            raise RuntimeError("R2 is not configured (R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY).")
        _client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )
    return _client


def presign_put(key: str, content_type: str, expires: int = 900) -> str:
    return client().generate_presigned_url(
        "put_object",
        Params={"Bucket": R2_BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
    )


def head(key: str) -> dict | None:
    try:
        return client().head_object(Bucket=R2_BUCKET, Key=key)
    except client().exceptions.ClientError:
        return None


def get_bytes(key: str) -> bytes:
    return client().get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
