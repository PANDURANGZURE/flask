import cv2
import easyocr
import json
import os
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__)

# Ensure owners.json exists
if not os.path.exists("owners.json"):
    sample_data = {
        "TNOZC3098": {"name": "Rahul Kumar", "phone": "9876543210"},
        "MH12AB1234": {"name": "Sneha Patil", "phone": "9123456780"}
    }
    with open("owners.json", "w") as f:
        json.dump(sample_data, f, indent=4)

# Load Owner Details
with open("owners.json", "r") as f:
    owners = json.load(f)

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

def generate_frames():
    """Stream camera frames to browser"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Capture current frame and run OCR"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Camera not available"}), 500

    success, frame = cap.read()
    cap.release()

    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    # --- Preprocessing for better OCR ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # Convert to grayscale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)          # Noise reduction
    edged = cv2.Canny(gray, 30, 200)                      # Edge detection

    # OCR expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = reader.readtext(rgb)

    print("OCR Raw Results:", results)  # Debugging

    detected = []
    for (_, text, _) in results:
        plate_number = text.replace(" ", "").upper()
        if plate_number in owners:
            detected.append({"plate": plate_number, "owner": owners[plate_number]})
        else:
            detected.append({"plate": plate_number, "owner": "Not Found"})

    if not detected:
        return jsonify({"error": "No text detected"}), 200

    return jsonify(detected)

if __name__ == '__main__':
    app.run(debug=True)
