# ocr_utils/extract_dl.py
from pathlib import Path
from PIL import Image
import io
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from .helpers import heuristic_name_split, clean_text, BASE_DIR

MODELS_DIR = BASE_DIR / "models"
FACE_MODELS_DIR = BASE_DIR / "face_models"

dl_model = YOLO(str(MODELS_DIR / "driving_licence.pt"))
ocr = PaddleOCR(use_gpu=False, lang='en', use_angle_cls=True)

dl_classes = [
    "Address", "Blood Group", "DL No", "DOB", "Name",
    "Relation With", "RTO", "State", "Vehicle Type"
]

face_net = cv2.dnn.readNetFromCaffe(str(FACE_MODELS_DIR / "deploy.prototxt"),
                                   str(FACE_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"))

def safe_ocr(roi):
    try:
        result = ocr.ocr(roi, cls=True)
        if not result:
            return ""
        lines = []
        for line in result:
            words = [w[1][0] for w in line]
            lines.append(" ".join(words))
        return " ".join(lines).strip()
    except:
        import pytesseract
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, config="--psm 6").strip()

def detect_face(roi):
    h, w = roi.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face_crop = roi[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            _, buf = cv2.imencode(".jpg", face_crop)
            return base64.b64encode(buf).decode()
    return None

def extract_driving_license_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = dl_model(cv_img)
    data = {"DL No": None, "Name": None, "DOB": None, "Relation With": None,
            "Address": None, "RTO": None, "State": None, "Vehicle Type": None,
            "Blood Group": None, "Portrait": None}

    name_detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = dl_classes[cls_id] if cls_id < len(dl_classes) else "Unknown"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = cv_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            text = safe_ocr(roi)
            text = clean_text(text)
            if label == "Name" and text:
                name_detections.append((y1, text))
            elif label in data:
                data[label] = text if text else None

    name_detections.sort(key=lambda x: x[0])
    if len(name_detections) > 0:
        all_names = [nd[1] for nd in name_detections]
        if len(all_names) == 1:
            data["Name"] = heuristic_name_split(all_names[0])
        elif len(all_names) >= 2:
            data["Name"] = {
                "first_name": all_names[1],
                "middle_name": " ".join(all_names[2:-1]) if len(all_names) > 2 else "",
                "last_name": all_names[0]
            }
        else:
            data["Name"] = heuristic_name_split(" ".join(all_names))

    data["Portrait"] = detect_face(cv_img)
    return data
