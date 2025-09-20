from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2, numpy as np, base64

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        img_b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        return jsonify({"emotion": result.get("dominant_emotion", "neutral")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

