from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64

app = Flask(__name__)
# Allow requests from your frontend origin (GitHub Pages/Netlify)
CORS(app, resources={r"/analyze": {"origins": "*"}})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing image data"}), 400

        # Decode base64 "data:image/jpeg;base64,...."
        img_b64 = data["image"]
        header, _, b64data = img_b64.partition(",")
        if not b64data:
            return jsonify({"error": "Invalid image data URL"}), 400

        img_bytes = base64.b64decode(b64data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze emotion; returns list of dicts
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        # DeepFace can return dict or list depending on version
        if isinstance(result, list):
            result = result[0]
        dominant = result.get('dominant_emotion', 'neutral')

        return jsonify({"emotion": dominant})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing only; host/port set by platform in production
    app.run(host="0.0.0.0", port=5000)
