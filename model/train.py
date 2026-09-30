"""Train the classifier on face embeddings and write server/artifacts.

Usage:  python model/train.py

Each photo in model/dataset/<person>/ is run through YuNet (largest face) and
SFace (128-d embedding); a classifier is trained on top. Photos are split by
source image, so a photo and its flipped copy never straddle train/test.
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from face_engine import FaceEngine  # noqa: E402

DATASET = ROOT / "model" / "dataset"
ARTIFACTS = ROOT / "server" / "artifacts"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_dataset(engine):
    classes = sorted(p.name for p in DATASET.iterdir() if p.is_dir() and p.name != "cropped")
    class_dict = {name: i for i, name in enumerate(classes)}
    X, y, groups = [], [], []
    for name in classes:
        used = skipped = 0
        for path in sorted((DATASET / name).iterdir()):
            if path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(path))
            if img is None:
                skipped += 1
                continue
            img, _ = engine.shrink(img)
            faces = engine.detect(img)
            if len(faces) == 0:
                skipped += 1
                continue
            # the subject is the largest face in a single-person photo
            face = faces[np.argmax(faces[:, 2] * faces[:, 3])]
            flipped = cv2.flip(img, 1)
            flipped_faces = engine.detect(flipped)
            variants = [engine.embed(img, face)]
            if len(flipped_faces):
                variants.append(engine.embed(flipped, flipped_faces[np.argmax(flipped_faces[:, 2] * flipped_faces[:, 3])]))
            for emb in variants:
                X.append(emb)
                y.append(class_dict[name])
                groups.append(f"{name}/{path.name}")
            used += 1
        print(f"{name:18s} used {used:3d}  skipped {skipped}")
    return np.array(X), np.array(y), np.array(groups), class_dict


def main():
    engine = FaceEngine()
    X, y, groups, class_dict = load_dataset(engine)
    names = sorted(class_dict, key=class_dict.get)
    print(f"\n{len(y)} embeddings (incl. flips), {len(names)} classes")

    candidates = {
        "logistic_regression": (LogisticRegression(max_iter=5000), {"C": [1, 10, 100, 1000]}),
        "svm": (SVC(probability=True, random_state=0), {"C": [1, 10, 100], "kernel": ["rbf", "linear"]}),
    }
    # grouped, stratified 5-fold: honest estimate, no flip/photo leakage
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
    best_name, best_est, best_score = None, None, -1
    for name, (model, params) in candidates.items():
        gs = GridSearchCV(model, params, cv=cv).fit(X, y)
        print(f"{name:20s} grouped-CV accuracy {gs.best_score_:.3f}  {gs.best_params_}")
        if gs.best_score_ > best_score:
            best_name, best_est, best_score = name, gs.best_estimator_, gs.best_score_

    # out-of-fold predictions for the report and to calibrate the "unknown" threshold
    oof = np.zeros((len(y), len(names)))
    for tr, te in cv:
        oof[te] = best_est.fit(X[tr], y[tr]).predict_proba(X[te])
    pred = oof.argmax(1)
    print(f"\nBest: {best_name}\n")
    print(classification_report(y, pred, target_names=names))
    print("confusion matrix:\n", confusion_matrix(y, pred))
    conf = oof.max(1)
    for t in (0.5, 0.6, 0.7, 0.8):
        keep = conf >= t
        print(f"threshold {t:.1f}: answers {keep.mean():.0%} of faces, accuracy when answering {(pred[keep] == y[keep]).mean():.3f}")

    # open-set check: mean cosine similarity of a face to its 3 nearest gallery
    # faces of the predicted class. Strangers get a probability too (the
    # classifier must pick one of five), but their similarity is low.
    def sim_to_class(q, gx, gy, c, k=3):
        return np.sort(gx[gy == c] @ q)[::-1][:k].mean()

    genuine, impostor = [], []
    for tr, te in cv:
        for i in te:
            genuine.append(sim_to_class(X[i], X[tr], y[tr], y[i]))
            impostor += [sim_to_class(X[i], X[tr], y[tr], c) for c in set(y) - {y[i]}]
    genuine, impostor = np.array(genuine), np.array(impostor)
    print(f"\nsimilarity to own class (genuine): median {np.median(genuine):.2f}, 10th pct {np.percentile(genuine, 10):.2f}")
    print(f"similarity to another person's class (impostor): median {np.median(impostor):.2f}, 95th pct {np.percentile(impostor, 95):.2f}")
    for t in (0.45, 0.5, 0.55):
        print(f"similarity threshold {t}: accepts {(genuine >= t).mean():.0%} of genuine, {(impostor >= t).mean():.0%} of impostors")

    best_est.fit(X, y)
    np.savez_compressed(ARTIFACTS / "gallery.npz", embeddings=X.astype(np.float32), labels=y)
    joblib.dump(best_est, ARTIFACTS / "saved_model.pkl", compress=3)
    (ARTIFACTS / "class_dictionary.json").write_text(json.dumps(class_dict))
    print("\nsaved to", ARTIFACTS)


if __name__ == "__main__":
    main()
