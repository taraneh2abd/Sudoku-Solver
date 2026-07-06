import os
import json
import time
import subprocess

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

UPLOAD_FOLDER = "app/uploads"

ORIGINAL_JSON = "results/00_original.json"
SOLVED_JSON = "results/00_solved.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("results", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return jsonify({"error": "No image selected"}), 400

    file = request.files["image"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # حذف خروجی‌های قبلی
    for f in [ORIGINAL_JSON, SOLVED_JSON]:
        if os.path.exists(f):
            os.remove(f)

    # اجرای solver
    process = subprocess.Popen(
        ["python", "main.py", filepath]
    )

    # منتظر ساخت original
    timeout = 30
    start = time.time()

    while not os.path.exists(ORIGINAL_JSON):
        if time.time() - start > timeout:
            process.kill()
            return jsonify({"error": "Timeout waiting for original board"}), 500

        time.sleep(0.2)

    with open(ORIGINAL_JSON, encoding="utf-8") as f:
        original = json.load(f)

    # منتظر solved
    while not os.path.exists(SOLVED_JSON):
        if time.time() - start > timeout:
            process.kill()
            return jsonify({"error": "Timeout waiting for solved board"}), 500

        time.sleep(0.2)

    with open(SOLVED_JSON, encoding="utf-8") as f:
        solved = json.load(f)

    return jsonify({
        "image": "/" + filepath.replace("\\", "/"),
        "original": original,
        "solved": solved
    })


@app.route("/app/uploads/<path:filename>")
def uploaded(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)