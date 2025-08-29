import cv2
import easyocr
import json
import os

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

print("Press 'c' to capture frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Feed", frame)  # Show live feed

    key = cv2.waitKey(1) & 0xFF

    # Capture frame and process
    if key == ord('c'):
        results = reader.readtext(frame)

        for (bbox, text, prob) in results:
            plate_number = text.replace(" ", "").upper()
            print("\nDetected Number Plate:", plate_number)

            # Lookup Owner Details
            if plate_number in owners:
                print("✅ Owner Found:", owners[plate_number])
            else:
                print("❌ Owner Not Found in Database")

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
