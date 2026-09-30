import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import util

UI_DIR = Path(__file__).resolve().parent.parent / "UI"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
CORS(app)

util.load_saved_artifacts()


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "app.html")


@app.route("/<path:filename>")
def ui_files(filename):
    return send_from_directory(UI_DIR, filename)


@app.route("/healthz")
def healthz():
    return jsonify(status="ok")


@app.route("/classify_image", methods=["POST"])
def classify_image():
    image_data = request.form.get("image_data")
    if not image_data:
        return jsonify(error="Missing 'image_data' field"), 400
    try:
        return jsonify(util.classify_image(image_data))
    except util.ImageError as e:
        return jsonify(error=str(e)), 400


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image too large (max 10 MB)"), 413


if __name__ == "__main__":
    print("Sports Celebrity Image Classification server: http://127.0.0.1:5001 (override with PORT)")
    app.run(port=int(os.environ.get("PORT", 5001)))
