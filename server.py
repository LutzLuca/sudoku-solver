from flask import Flask, request, jsonify, render_template
import sys
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import cv2 as cv

from sudoku_solver.main import process_sudoku_image

app = Flask(__name__)
model = None

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def solve_sudoku():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    img = request.files["file"]

    try:
        stream = io.BytesIO(img.read())
        img = Image.open(stream)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img)
        grayscaled = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)

        solved_grid = process_sudoku_image(model, grayscaled)
        return jsonify({"success": True, "solved_grid": solved_grid.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        print("Usage: python server.py <path to a digit recognition model>")
        exit(1)

    model = tf.keras.models.load_model(argv[1])
    assert model is not None

    app.run(debug=True, host="0.0.0.0", port=8000)
