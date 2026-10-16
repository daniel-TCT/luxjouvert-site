"""Face detection + fingerprinting with OpenCV's YuNet (MIT) and SFace (Apache-2.0).

Both models are free for commercial use and run on CPU with no compiling,
so the Railway build stays fast and cheap.
"""
import hashlib
import os
import threading
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/lux-models")
_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS = {
    "det": ("face_detection_yunet_2023mar.onnx",
            f"{_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "rec": ("face_recognition_sface_2021dec.onnx",
            f"{_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}

# Selfie quality gates
MIN_FACE_PX = 120          # face box must be at least this wide in the selfie
MIN_DET_SCORE = 0.90       # detector confidence
MIN_SHARPNESS = 60.0       # Laplacian variance on the aligned 112x112 face; sharp ≈ 600+, blurry ≈ 10-25

# Matching threshold on L2 distance between unit-length fingerprints.
# SFace's published cosine threshold is 0.363 → L2 ≈ sqrt(2 - 2*0.363) ≈ 1.128.
MATCH_MAX_DISTANCE = float(os.getenv("MATCH_MAX_DISTANCE", "1.128"))

_lock = threading.Lock()
_det = None
_rec = None


def _ensure(name: str) -> str:
    fname, url, sha = MODELS[name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if digest != sha:
            os.remove(tmp)
            raise RuntimeError(f"Checksum mismatch for {fname}")
        os.replace(tmp, path)
    return path


def load_models():
    global _det, _rec
    with _lock:
        if _det is None:
            _det = cv2.FaceDetectorYN.create(_ensure("det"), "", (320, 320), 0.6, 0.3, 5000)
        if _rec is None:
            _rec = cv2.FaceRecognizerSF.create(_ensure("rec"), "")


class FaceError(ValueError):
    """A user-facing reason the photo can't be used."""


def decode_image(data: bytes, max_side: int = 1600) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("That file isn't a readable image.")
    # Cap size so huge camera files don't eat memory/CPU
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect(img: np.ndarray) -> np.ndarray:
    load_models()
    h, w = img.shape[:2]
    with _lock:
        _det.setInputSize((w, h))
        _, faces = _det.detect(img)
    return faces if faces is not None else np.empty((0, 15), np.float32)


def embed(img: np.ndarray, face_row: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit-length fingerprint, sharpness of the aligned 112x112 face)."""
    load_models()
    with _lock:
        aligned = _rec.alignCrop(img, face_row)
        feat = _rec.feature(aligned).flatten().astype(np.float32)
    # Sharpness on the fixed-size aligned crop, so it doesn't depend on camera resolution
    sharp = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    n = np.linalg.norm(feat)
    return (feat / n if n else feat), sharp


def selfie_embedding(data: bytes) -> list[float]:
    """Validate a selfie and return its 128-number fingerprint."""
    img = decode_image(data)
    faces = detect(img)
    faces = faces[faces[:, 14] >= MIN_DET_SCORE] if len(faces) else faces
    if len(faces) == 0:
        raise FaceError("We couldn't find a face. Face the camera in good light.")
    if len(faces) > 1:
        raise FaceError("More than one face is in the photo. Take it on your own.")
    f = faces[0]
    if int(f[2]) < MIN_FACE_PX:
        raise FaceError("Your face is too small in the frame. Move closer.")
    feat, sharp = embed(img, f)
    if sharp < MIN_SHARPNESS:
        raise FaceError("The photo is blurry. Hold still and try again.")
    return feat.tolist()


# Event-photo gates: looser than selfies, since crowd shots have smaller faces
EVENT_MAX_SIDE = 2400       # keep enough resolution for faces in group shots
EVENT_MIN_FACE_PX = 40
EVENT_MIN_DET_SCORE = 0.80
EVENT_MIN_SHARPNESS = 30.0


def photo_faces(data: bytes) -> tuple[list[dict], float]:
    """All usable faces in an event photo → ([{bbox, embedding, sharpness}], photo quality)."""
    img = decode_image(data, EVENT_MAX_SIDE)
    out, sharps = [], []
    for f in detect(img):
        if f[14] < EVENT_MIN_DET_SCORE or f[2] < EVENT_MIN_FACE_PX:
            continue
        feat, sharp = embed(img, f)
        sharps.append(sharp)
        if sharp < EVENT_MIN_SHARPNESS:
            continue
        x, y, w, h = (float(v) for v in f[:4])
        out.append({"bbox": {"x": x, "y": y, "w": w, "h": h, "score": float(f[14])},
                    "embedding": feat.tolist(), "sharpness": sharp})
    quality = float(np.median(sharps)) if sharps else 0.0
    return out, quality


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))
