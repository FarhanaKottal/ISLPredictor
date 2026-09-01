# Indian Sign Language Predictor using YOLO26

A real-time **Indian Sign Language (ISL) recognition system** that uses a webcam, a trained YOLO26 object-detection model, and a Flask web application to recognize predefined hand gestures.

The system detects an ISL gesture from an image captured by the user's camera, predicts the corresponding gesture with a confidence score, and sends the result to the web interface. The frontend can then display the recognized gesture and convert the predicted result into speech using text-to-speech.

---

## 📌 Project Overview

Communication can be challenging for people who use sign language when the person they are communicating with does not understand sign language.

This project aims to provide a simple computer-vision-based communication interface by recognizing selected Indian Sign Language gestures through a camera.

The system uses:

**Webcam → Image → YOLO26 Model → Gesture Prediction → Text → Speech**

The current model recognizes the following six gestures:

* `Hello`
* `IloveYou`
* `No`
* `Please`
* `Thanks`
* `Yes`

For example:

> User performs the **Hello** sign → Camera captures the image → YOLO26 detects the gesture → Application predicts **Hello** → The result is displayed and can be spoken aloud.

---

## ✨ Features

* Real-time camera-based gesture recognition
* YOLO26 object-detection model
* Flask backend
* REST-style `/predict` endpoint
* Confidence threshold of **80%**
* Six predefined ISL gestures
* Returns prediction and confidence as JSON
* Frontend integration for displaying predictions
* Text-to-speech support through the frontend
* GPU-based model training using Google Colab

---

## 🧠 Supported Gestures

| Gesture  | Description                  |
| -------- | ---------------------------- |
| Hello    | ISL gesture for "Hello"      |
| IloveYou | ISL gesture for "I Love You" |
| No       | ISL gesture for "No"         |
| Please   | ISL gesture for "Please"     |
| Thanks   | ISL gesture for "Thanks"     |
| Yes      | ISL gesture for "Yes"        |

The application intentionally accepts only these six classes. If the YOLO model detects another class, it is ignored by the Flask prediction logic.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │     Webcam      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Capture Image  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Flask Frontend  │
                    │   /predict API  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    YOLO26       │
                    │ Trained Model   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Gesture +       │
                    │ Confidence      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Web Interface   │
                    │ Display Result  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text-to-Speech  │
                    │    Speaker      │
                    └─────────────────┘
```

---

## 🔬 Machine Learning Model

The project uses **YOLO26** for object detection.

The model was trained using the Ultralytics YOLO framework. The training notebook uses Google Colab with GPU acceleration.

The trained model is saved as:

```text
best.pt
```

The Flask application loads the model using:

```python
model = YOLO("best.pt")
```

The model performs object detection on each submitted image and selects the detection with the highest confidence.

---

## 📊 Model Training

The model was trained for:

```text
100 epochs
```

The training was performed using a YOLO26m model.

According to the training results:

```text
YOLO26m
132 layers
20,354,078 parameters
67.9 GFLOPs
```

The training environment reported:

```text
Ultralytics 8.4.96
Python 3.12.13
PyTorch 2.11.0+cu128
GPU: NVIDIA Tesla T4
```

The training completed in approximately:

```text
0.253 hours
```

---

## 📈 Model Performance

The final validation was performed on a test set containing **30 images**, with 5 images for each of the six classes.

Overall validation results:

| Metric    | Result |
| --------- | -----: |
| Precision |  0.986 |
| Recall    |  0.998 |
| mAP50     |  0.995 |
| mAP50-95  |  0.917 |

### Per-Class Results

| Class    | Images | Precision | Recall | mAP50 | mAP50-95 |
| -------- | -----: | --------: | -----: | ----: | -------: |
| Hello    |      5 |     1.000 |  0.987 | 0.995 |    0.947 |
| IloveYou |      5 |     1.000 |  1.000 | 0.995 |    0.913 |
| No       |      5 |     0.950 |  1.000 | 0.995 |    0.905 |
| Please   |      5 |     0.993 |  1.000 | 0.995 |    0.899 |
| Thanks   |      5 |     1.000 |  1.000 | 0.995 |    0.911 |
| Yes      |      5 |     0.974 |  1.000 | 0.995 |    0.924 |

These are the results recorded in the project notebook.

> **Note:** The validation dataset is relatively small (30 images), so these metrics should not be interpreted as proof of real-world accuracy under all lighting conditions, backgrounds, users, camera angles, and distances.

---

# 💻 Flask Application

The backend is implemented using Flask.

The main application:

```text
app.py
```

### Main dependencies

```python
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
```

The Flask application:

1. Loads `best.pt`
2. Serves the web interface
3. Receives an image from the frontend
4. Converts the uploaded image into an OpenCV image
5. Runs YOLO inference
6. Finds the highest-confidence detection
7. Checks whether the detected class is one of the supported gestures
8. Returns the gesture and confidence as JSON

---

## 🔌 API

### `GET /`

Loads the main web interface.

```text
/
```

The route renders:

```text
templates/index.html
```

---

### `POST /predict`

Receives an image and performs gesture recognition.

```text
/predict
```

The image must be sent using the form field:

```text
image
```

### Example response

If the system detects the `Hello` gesture:

```json
{
    "gesture": "Hello",
    "confidence": 96.42
}
```

If no valid gesture is detected:

```json
{
    "gesture": "",
    "confidence": 0
}
```

---

# 🎯 Confidence Threshold

The application uses:

```python
CONFIDENCE_THRESHOLD = 0.80
```

Therefore, YOLO detections below 80% confidence are not considered.

For example:

```text
Confidence = 0.92
```

becomes:

```text
92.0%
```

and can be returned as a valid prediction.

---

# 🔊 Text-to-Speech

The Flask backend returns the recognized gesture to the frontend.

For example:

```json
{
    "gesture": "Thanks",
    "confidence": 94.56
}
```

The frontend can use this result to:

1. Display `Thanks`
2. Convert `Thanks` to speech
3. Play the speech through the computer's speaker

The speech-generation portion is handled by the client-side application rather than the `app.py` code shown here.

This separation allows the Flask backend to focus on computer vision and prediction while the frontend handles the user interface and speech output.

---

# 📁 Recommended Project Structure

```text
ISL-Predictor/
│
├── app.py
├── best.pt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    ├── script.js
    └── style.css
```

### File descriptions

| File                   | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `app.py`               | Flask server and YOLO prediction API          |
| `best.pt`              | Trained YOLO26 model                          |
| `templates/index.html` | Web interface                                 |
| `static/script.js`     | Camera handling, API communication and speech |
| `static/style.css`     | Interface styling                             |
| `README.md`            | Project documentation                         |

---

# ⚙️ Installation

## 1. Clone the project

```bash
git clone <repository-url>
cd ISL-Predictor
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

### Linux/macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install flask ultralytics numpy opencv-python
```

---

## 4. Add the trained model

Place the trained model in the project directory:

```text
ISL-Predictor/
├── app.py
├── best.pt
└── ...
```

The application expects:

```python
model = YOLO("best.pt")
```

---

# ▶️ Running the Application

Start the Flask server:

```bash
python app.py
```

The application will start the Flask development server.

Open the address displayed by Flask in your web browser.

For example:

```text
http://127.0.0.1:5000
```

Allow the browser to access the webcam when prompted.

---

# 🔄 Prediction Workflow

When the user performs an ISL gesture:

### Step 1 — Camera

The webcam captures an image containing the hand gesture.

### Step 2 — Upload

The frontend sends the captured image to:

```text
POST /predict
```

### Step 3 — Image Processing

Flask receives the image and converts the uploaded bytes into an OpenCV image:

```python
image_bytes = np.frombuffer(file.read(), np.uint8)
image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
```

### Step 4 — YOLO Prediction

The image is passed to the trained YOLO model:

```python
results = model.predict(
    source=image,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)
```

### Step 5 — Select Best Detection

If multiple objects are detected, the application selects the detection with the highest confidence.

```python
best_box = max(
    result.boxes,
    key=lambda b: float(b.conf[0])
)
```

### Step 6 — Validate Gesture

The predicted class is compared against:

```python
VALID_GESTURES = [
    "Hello",
    "IloveYou",
    "No",
    "Please",
    "Thanks",
    "Yes"
]
```

Only these gestures are accepted.

### Step 7 — Return Result

The server returns:

```json
{
    "gesture": "Hello",
    "confidence": 95.21
}
```

### Step 8 — Speech

The frontend receives the prediction and can speak the recognized gesture using text-to-speech.

---

# 🧪 Model Testing

The trained model can also be tested directly using Ultralytics.

Example:

```bash
yolo predict model=best.pt source=<image-or-folder>
```

Validation can be performed using:

```bash
yolo val model=best.pt data=data.yaml
```

The original training notebook contains prediction and validation experiments using the trained `best.pt` model.

---

# 📚 Dataset

The project uses images representing the six supported gestures.

The training notebook shows images organized into classes including:

```text
Hello
IloveYou
No
Please
Thanks
Yes
```

The dataset was used for training and evaluation of the YOLO26 model. The recorded validation set contains 30 images, with five instances per class.

For a more robust real-world system, a significantly larger and more diverse dataset would be recommended.

---

# 🚀 Future Improvements

Possible improvements include:

* Add more Indian Sign Language gestures
* Increase the size and diversity of the dataset
* Support multiple users
* Improve recognition under different lighting conditions
* Add background variation during training
* Improve temporal stability for video recognition
* Add gesture-history support
* Convert multiple gestures into complete sentences
* Add multilingual speech output
* Add offline text-to-speech
* Deploy the model on an edge device such as NVIDIA Jetson
* Optimize the model for faster inference
* Add voice-to-sign functionality
* Add a mobile application

---

# ⚠️ Limitations

The current implementation recognizes only six predefined gestures:

```text
Hello
IloveYou
No
Please
Thanks
Yes
```

It is therefore **not a complete Indian Sign Language translation system**.

The model is trained for a limited set of static gesture classes and should not be described as a general-purpose ISL translator.

The validation dataset used in the notebook is also small, containing 30 test images. Real-world performance may therefore differ from the reported validation metrics.

---

# 🛠️ Technologies Used

* **Python**
* **Flask**
* **Ultralytics YOLO26**
* **OpenCV**
* **NumPy**
* **HTML/CSS/JavaScript**
* **Webcam**
* **Text-to-Speech**
* **Google Colab**
* **GPU acceleration**

---

# 📜 License

This project is intended for educational and research purposes.

If you redistribute or modify the project, make sure to comply with the licenses of the third-party libraries and models used by the project.

---

# 👩‍💻 Project Summary

This project demonstrates how computer vision and deep learning can be used to build an assistive communication system for Indian Sign Language.

A trained YOLO26 model recognizes six predefined hand gestures from camera images. A Flask backend provides an API for performing inference, while the web frontend displays the recognized gesture and can convert the prediction into spoken output.

The overall concept is:

```text
Indian Sign Language
        ↓
     Webcam
        ↓
     YOLO26
        ↓
Gesture Recognition
        ↓
      Text
        ↓
 Text-to-Speech
        ↓
     Speaker
```

The project provides a foundation that can be extended into a larger ISL recognition and communication system.
