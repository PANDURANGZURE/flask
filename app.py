import cv2
import easyocr
import json
import os
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__)

# Ensure owners.json exists with sample data
if not os.path.exists("owners.json"):
    sample_data = {
        "KA20.1025": {"name": "Rahul Kumar", "phone": "9876543210"},
        "MH12AB1234": {"name": "Sneha Patil", "phone": "9123456780"}
    }
    with open("owners.json", "w") as f:
        json.dump(sample_data, f, indent=4)

# Load Owner Details
with open("owners.json", "r") as f:
    owners = json.load(f)

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

# Open Camera
cap = cv2.VideoCapture(0)


def generate_frames():
    """Stream camera frames to browser"""
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')  # HTML page with video


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    """Capture current frame and run OCR"""
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    results = reader.readtext(frame)
    detected = []
    for (_, text, _) in results:
        plate_number = text.replace(" ", "").upper()
        if plate_number in owners:
            detected.append({"plate": plate_number, "owner": owners[plate_number]})
        else:
            detected.append({"plate": plate_number, "owner": "Not Found"})

    return jsonify(detected)


if __name__ == '__main__':
    app.run(debug=True)
