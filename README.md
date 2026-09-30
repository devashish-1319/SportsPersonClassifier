# Athlete Vision — Sports Person Classifier

Upload a photo and the app finds every face and tells you which of five athletes
(Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, Virat Kohli) it is,
with a confidence score per face. Faces it doesn't recognise are reported as **unknown**.

## How it works

1. **Detect** – YuNet (OpenCV DNN) finds every face.
2. **Embed** – SFace turns each aligned face into a 128-d vector.
3. **Classify** – an SVM on the embeddings gives per-athlete probabilities.
4. **Reject strangers** – a face is only named if its probability is ≥ 60% *and* it is close
   (cosine similarity ≥ 0.5) to known photos of that athlete.

Cross-validated accuracy is **~95%** (grouped by source photo, so no leakage), and ~98–99% on the faces it
commits to. The previous Haar-cascade + pixel/wavelet model scored ~81%.
Run `python model/train.py` for the full report. `sports_person_classifier_model.ipynb` walks through it.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server/server.py            # http://127.0.0.1:5001  (set PORT to change)
```

## Deploy

The app is a single Flask service that also serves the UI.

```bash
docker build -t athlete-vision .
docker run -p 8000:8000 athlete-vision     # http://localhost:8000
```

On Render / Railway / Fly / Heroku-style hosts, point them at the repo: the `Dockerfile` (or `Procfile`) is
enough. Health check: `GET /healthz`. Needs ~300 MB RAM.

## Retrain

```bash
python model/train.py      # reads model/dataset/<person>/, writes server/artifacts/
```

To add an athlete, drop photos into `model/dataset/<name>/`, add a photo to `UI/images/` and an entry in
`PLAYERS` in `UI/app.js`, then retrain.

## API

`POST /classify_image` with form field `image_data` (base64 data URL) returns

```json
{"faces": [{"class": "virat_kohli", "confidence": 98.8, "known": true, "similarity": 0.71,
            "probabilities": {"virat_kohli": 98.8, "...": 0.3}, "box": [x, y, w, h]}],
 "image_size": [width, height]}
```

Errors return `{"error": "..."}` with HTTP 4xx. `known: false` means the face wasn't confidently matched.

## Limits

Trained on a small set of public photos, for learning purposes. Don't use it to identify private individuals.
Models: YuNet and SFace from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo).
