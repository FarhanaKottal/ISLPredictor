from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# Load your trained model
model = YOLO("best.pt")

# Only these gestures are valid
VALID_GESTURES = [
    "Hello",
    "IloveYou",
    "No",
    "Please",
    "Thanks",
    "Yes"
]

CONFIDENCE_THRESHOLD = 0.80


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "gesture": "",
            "confidence": 0
        })

    file = request.files["image"]

    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    gesture = ""
    confidence = 0

    if len(results) > 0:

        result = results[0]

        if len(result.boxes) > 0:

            # Pick the detection with highest confidence
            best_box = max(
                result.boxes,
                key=lambda b: float(b.conf[0])
            )

            confidence = float(best_box.conf[0])

            class_id = int(best_box.cls[0])

            predicted = model.names[class_id]

            if predicted in VALID_GESTURES:
                gesture = predicted

    return jsonify({
        "gesture": gesture,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)