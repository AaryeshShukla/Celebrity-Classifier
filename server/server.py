import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_from_directory
import util

app = Flask(__name__, static_folder=None)

util.load_saved_artifacts()


@app.route("/")
def index():
    return send_from_directory(app.root_path, "index.html")


@app.route("/app.js")
def app_js():
    return send_from_directory(app.root_path, "app.js")


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(os.path.join(BASE_DIR, "images"), filename)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.form.get('image_data')   # frontend sends base64
    if not image_data:
        return jsonify({"error": "image_data is required"}), 400

    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Flask server for Celebrity Classifier...")
    app.run(host="127.0.0.1", port=5000, debug=True)
