"""Face detection (YuNet) and embedding (SFace) via OpenCV's DNN module.

Shared by training (model/train.py) and serving (server/util.py).
"""
from pathlib import Path

import cv2
import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DETECTOR_PATH = ARTIFACTS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = ARTIFACTS_DIR / "face_recognition_sface_2021dec.onnx"

MAX_SIDE = 1280  # downscale huge photos before detection


class FaceEngine:
    def __init__(self, score_threshold=0.8):
        self._detector = cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (320, 320), score_threshold, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")

    @staticmethod
    def shrink(img):
        """Return (img, scale) with the longest side capped at MAX_SIDE."""
        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img, scale

    def detect(self, img):
        """Return an (n, 15) array of YuNet detections (box, 5 landmarks, score)."""
        h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img)
        return np.empty((0, 15), np.float32) if faces is None else faces

    def embed(self, img, face_row):
        """L2-normalised 128-d embedding of an aligned face."""
        aligned = self._recognizer.alignCrop(img, face_row)
        feat = self._recognizer.feature(aligned).ravel()
        return feat / (np.linalg.norm(feat) + 1e-12)
