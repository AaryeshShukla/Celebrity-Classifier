from flask import Flask, request, jsonify
import util

app = Flask(__name__)

# Route for image classification
@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        image_data = request.form['image_data']
        result = util.classify_image(image_base64_data=image_data)
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Starting Python Flask Server For Celebrity Image Classification...")
    util.load_saved_artifacts()
    app.run(host="127.0.0.1", port=5000, debug=True)
