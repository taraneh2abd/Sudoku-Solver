import os
import json
import time
import uuid
import subprocess

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

UPLOAD_FOLDER = "app/uploads"
RESULTS_ROOT = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

UPLOAD_TIMEOUT_SECONDS = 30


@app.route("/")
def index():
    return render_template("index.html")


def _wait_for_file(path, process, timeout):
    """Poll for `path` to appear. Returns None on success, or an error
    string if the wait timed out or the subprocess exited early without
    producing the file (in which case we don't wait out the rest of the
    timeout - the process is already gone, so surface its stderr instead)."""
    start = time.time()

    while not os.path.exists(path):
        if process.poll() is not None:
            _, stderr = process.communicate()
            detail = stderr.strip() if stderr else f"exit code {process.returncode}"
            return f"main.py failed before writing {os.path.basename(path)}: {detail}"

        if time.time() - start > timeout:
            process.kill()
            return f"Timeout waiting for {os.path.basename(path)}"

        time.sleep(0.2)

    return None


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return jsonify({"error": "No image selected"}), 400

    file = request.files["image"]

    # every upload gets its own id: previously the upload filename and the
    # results/*.json paths were fixed/global, so two concurrent uploads
    # (two tabs, two users) would silently overwrite each other's files.
    job_id = uuid.uuid4().hex
    filename = f"{job_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    output_dir = os.path.join(RESULTS_ROOT, job_id)
    original_json = os.path.join(output_dir, "00_original.json")
    solved_json = os.path.join(output_dir, "00_solved.json")

    process = subprocess.Popen(
        ["python", "main.py", filepath, output_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # each wait gets its own fresh timeout budget (previously a single
    # `start` timestamp was shared across both waits, so the true combined
    # budget was 30s total instead of the apparent 30s-per-stage)
    error = _wait_for_file(original_json, process, UPLOAD_TIMEOUT_SECONDS)
    if error:
        return jsonify({"error": error}), 500

    with open(original_json, encoding="utf-8") as f:
        original = json.load(f)

    error = _wait_for_file(solved_json, process, UPLOAD_TIMEOUT_SECONDS)
    if error:
        return jsonify({"error": error}), 500

    with open(solved_json, encoding="utf-8") as f:
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