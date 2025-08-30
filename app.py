import os, json, re
import cv2
import numpy as np
import easyocr
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

# -------------------- Owners DB (normalize keys) --------------------
def normalize_plate(s: str) -> str:
    # Keep only A–Z and 0–9, uppercase
    return re.sub(r'[^A-Z0-9]', '', s.upper())

if not os.path.exists("owners.json"):
    sample_data = {
        "ET0705": {"name": "Rahul Kumar", "phone": "9876543210"},
        "MH12AB1234": {"name": "Sneha Patil", "phone": "9123456780"}
    }
    with open("owners.json", "w") as f:
        json.dump(sample_data, f, indent=4)

with open("owners.json", "r") as f:
    owners_raw = json.load(f)

# Build normalized lookup so "KA20 1025" or "KA20.1025" both match
owners = {}
for k, v in owners_raw.items():
    owners[normalize_plate(k)] = v

# -------------------- OCR --------------------
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA
ALLOW = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Indian plate patterns (common formats)
PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'),   # e.g. MH12AB1234 / KA01AA1234
    re.compile(r'^[A-Z]{2}\d{1}[A-Z]{1,2}\d{4}$'),   # e.g. rare single digit district
]

def is_plate_like(text: str) -> bool:
    t = normalize_plate(text)
    if len(t) < 7 or len(t) > 12:
        return False
    return any(p.fullmatch(t) for p in PLATE_PATTERNS)

# -------------------- Plate region detection --------------------
def find_plate_rois(frame_bgr):
    """Return list of (roi_rgb, bbox_xyxy) candidates likely to be plates."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Emphasize horizontal strokes (plates are wider)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad_x)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold + close to connect characters
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < (h * w) * 0.001 or area > (h * w) * 0.15:
            continue
        ar = cw / float(ch + 1e-6)
        if 2.0 <= ar <= 6.5:  # plate-ish aspect ratio
            # Expand a bit to include full plate
            pad = int(0.06 * max(cw, ch))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(w, x + cw + pad); y1 = min(h, y + ch + pad)
            roi = frame_bgr[y0:y1, x0:x1]
            if roi.size == 0: 
                continue
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            candidates.append((roi_rgb, (int(x0), int(y0), int(x1), int(y1))))
    # Sort by area desc (try larger first)
    candidates.sort(key=lambda t: (t[0].shape[0] * t[0].shape[1]), reverse=True)
    return candidates[:6]  # top few

def preprocess_variants(img_rgb):
    """Yield a few resized/thresholded variants to improve OCR."""
    outs = []

    # Base
    base = img_rgb.copy()
    outs.append(base)

    # Upscale
    up = cv2.resize(img_rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    outs.append(up)

    # Grayscale + CLAHE + Otsu
    g = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB))

    # Adaptive threshold
    ad = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    outs.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2RGB))

    return outs

def ocr_best_plate(img_rgb):
    """Run OCR with allowlist + postprocess; return (plate, conf, bbox_local)."""
    best = None
    for variant in preprocess_variants(img_rgb):
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist=ALLOW,
            decoder='greedy'
        )
        for (bbox, text, prob) in results:
            t = normalize_plate(text)
            if not t:
                continue
            # Prefer regex-valid plates; otherwise consider strong long tokens
            score = prob * (1.2 if is_plate_like(t) else 1.0) * min(len(t), 10)
            if best is None or score > best[1]:
                best = (t, score, prob, bbox)
    if best:
        return best[0], best[2], best[3]  # plate, prob, bbox
    return None, None, None

# -------------------- Streaming --------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- Capture & OCR --------------------
@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Camera not available"}), 500
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return jsonify({"error": "Failed to capture frame"}), 500

    # 1) Try plate ROIs first
    rois = find_plate_rois(frame)
    detections = []
    seen = set()

    def add_detection(plate, conf, bbox_img, bbox_frame=None):
        if not plate:
            return
        plate_norm = normalize_plate(plate)
        if plate_norm in seen:
            return
        seen.add(plate_norm)
        owner = owners.get(plate_norm, "Not Found")
        detections.append({
            "plate": plate_norm,
            "confidence": round(float(conf or 0), 3),
            "bbox": bbox_frame,  # [x0,y0,x1,y1] in full frame if available
            "owner": owner
        })

    for roi_rgb, (x0, y0, x1, y1) in rois:
        plate, conf, _bbox_local = ocr_best_plate(roi_rgb)
        add_detection(plate, conf, None, [x0, y0, x1, y1])

    # 2) Fallback: whole frame OCR if nothing from ROIs
    if not detections:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plate, conf, _bbox_local = ocr_best_plate(img_rgb)
        add_detection(plate, conf, _bbox_local, None)

    # 3) Filter to regex-valid plates first; if none, return whatever we got
    valid = [d for d in detections if is_plate_like(d["plate"])]
    result = valid if valid else detections

    if not result:
        return jsonify({"error": "No plate-like text detected"}), 200

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
