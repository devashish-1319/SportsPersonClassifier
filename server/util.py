import base64
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from face_engine import FaceEngine

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Below this confidence we report "unknown" rather than guess (chosen from
# out-of-fold predictions in model/train.py: ~99% accurate when above it).
UNKNOWN_THRESHOLD = 60.0
# Minimum mean cosine similarity to the 3 closest gallery faces of the predicted
# athlete. Rejects strangers the closed-set classifier would otherwise force
# into one of the five classes (see model/train.py for the calibration).
MIN_SIMILARITY = 0.5

__class_number_to_name = {}
__model = None
__engine = None
__gallery = None


class ImageError(ValueError):
    """Raised when the submitted image cannot be decoded."""


def load_saved_artifacts():
    global __class_number_to_name, __model, __engine, __gallery
    print("loading saved artifacts...start")
    with open(ARTIFACTS_DIR / "class_dictionary.json") as f:
        __class_number_to_name = {v: k for k, v in json.load(f).items()}
    if __model is None:
        __model = joblib.load(ARTIFACTS_DIR / "saved_model.pkl")
    if __gallery is None:
        g = np.load(ARTIFACTS_DIR / "gallery.npz")
        __gallery = (g["embeddings"], g["labels"])
    if __engine is None:
        __engine = FaceEngine()
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    """Decode a data URL (or bare base64 string) into a BGR image."""
    try:
        encoded = b64str.split(",", 1)[-1]
        buf = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ImageError("Could not decode image data") from e
    if img is None:
        raise ImageError("Data is not a valid image")
    return img


def class_similarity(emb, class_number, k=3):
    gx, gy = __gallery
    return float(np.sort(gx[gy == class_number] @ emb)[::-1][:k].mean())


def classify_image(image_base64_data=None, file_path=None):
    img = cv2.imread(file_path) if file_path else get_cv2_image_from_base64_string(image_base64_data)
    if img is None:
        raise ImageError("Could not read image")

    img, _ = __engine.shrink(img)
    height, width = img.shape[:2]
    results = []
    for face in __engine.detect(img):
        emb = __engine.embed(img, face)
        probs = __model.predict_proba(emb.reshape(1, -1))[0]
        best = int(np.argmax(probs))
        similarity = class_similarity(emb, int(__model.classes_[best]))
        confidence = round(float(probs[best]) * 100, 2)
        x, y, w, h = (max(0, int(v)) for v in face[:4])
        results.append({
            "class": __class_number_to_name[int(__model.classes_[best])],
            "confidence": confidence,
            "similarity": round(similarity, 3),
            "known": confidence >= UNKNOWN_THRESHOLD and similarity >= MIN_SIMILARITY,
            "probabilities": {
                __class_number_to_name[int(c)]: round(float(p) * 100, 2)
                for c, p in zip(__model.classes_, probs)
            },
            "box": [x, y, w, h],
        })
    results.sort(key=lambda r: r["box"][2] * r["box"][3], reverse=True)
    return {"faces": results, "image_size": [width, height]}


if __name__ == "__main__":
    load_saved_artifacts()
    for name in sorted((BASE_DIR / "test_images").glob("*.jpg")):
        out = classify_image(file_path=str(name))
        print(name.name, [(f["class"], f["confidence"]) for f in out["faces"]])
