# ocr_utils/extract_voter.py
from pathlib import Path
from PIL import Image
import io
import base64
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from .helpers import heuristic_name_split, clean_text, BASE_DIR

MODELS_DIR = BASE_DIR / "models"
FACE_MODELS_DIR = BASE_DIR / "face_models"

voter_model = YOLO(str(MODELS_DIR / "voter_id.pt"))
face_net = cv2.dnn.readNetFromCaffe(str(FACE_MODELS_DIR / "deploy.prototxt"),
                                   str(FACE_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"))

ocr = PaddleOCR(use_gpu=False, lang='en', use_angle_cls=True,
                enable_mkldnn=False, use_textline_orientation=True,
                rec_algorithm='SVTR_LCNet', det_algorithm='DB', ocr_version='PP-OCRv4')

VOTER_CLASSES = [
    "Address", "Age", "DOB", "Card Voter ID 1 Back", "Card Voter ID 2 Front",
    "Card Voter ID 2 Back", "Card Voter ID 1 Front", "DOB", "Date of Issue",
    "Election", "Father", "Gender", "Name", "Point", "Portrait", "Symbol",
    "Voter ID"
]
REMOVE_FIELDS = ["Card Voter ID 1 Front", "Election", "Symbol"]
voter_regex = re.compile(r'[A-Z]{3}[0-9]{7}')

def preprocess_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    gray = cv2.resize(gray, (w*2, h*2))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def preprocess_voter_id_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = threshed.shape
    if h == 0:
        return cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
    scale = 100 / h
    threshed = cv2.resize(threshed, (int(w*scale), 100))
    return cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

def safe_ocr_read(roi):
    try:
        result = ocr.ocr(roi, cls=True)
        if not result:
            return ""
        lines = []
        for line in result:
            words = [word_info[1][0] for word_info in line]
            lines.append(" ".join(words))
        return " ".join(lines).strip()
    except:
        import pytesseract
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, config="--psm 6").strip()

def detect_face_encode(roi):
    h, w = roi.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            face_crop = roi[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if face_crop.size == 0:
                continue
            _, buffer = cv2.imencode('.jpg', face_crop)
            return base64.b64encode(buffer).decode('utf-8')
    return None

def extract_voter_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = voter_model(cv_image)
    detected_data = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = VOTER_CLASSES[cls_id] if cls_id < len(VOTER_CLASSES) else f"Unknown_{cls_id}"
            if class_name in REMOVE_FIELDS or class_name.lower() == "portrait":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = cv_image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_proc = preprocess_voter_id_roi(roi) if "voterid" in class_name.lower() else preprocess_image(roi)
            text = safe_ocr_read(roi_proc)
            if "voterid" in class_name.lower() and not text:
                import pytesseract
                gray = cv2.cvtColor(roi_proc, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            if "voterid" in class_name.lower():
                match = voter_regex.search(text.replace(" ", ""))
                text = match.group(0) if match else text
            if class_name.lower() == "name":
                detected_data["Name"] = heuristic_name_split(text)
            elif class_name.lower() == "father":
                detected_data["Father"] = text.title() if text else None
            else:
                detected_data[class_name] = text if text else None

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = VOTER_CLASSES[cls_id] if cls_id < len(VOTER_CLASSES) else f"Unknown_{cls_id}"
            if class_name.lower() != "portrait":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = cv_image[y1:y2, x1:x2]
            detected_data["Portrait"] = detect_face_encode(roi)

    for key in ["Name", "Father", "Portrait"]:
        if key not in detected_data:
            detected_data[key] = None

    return detected_data
