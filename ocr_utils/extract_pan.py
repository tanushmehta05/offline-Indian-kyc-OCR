# ocr_utils/extract_pan.py
from pathlib import Path
from PIL import Image
import io
import base64
import re
import cv2
import numpy as np
from ultralytics import YOLO
from .helpers import heuristic_name_split, clean_text, BASE_DIR

MODELS_DIR = BASE_DIR / "models"
FACE_MODELS_DIR = BASE_DIR / "face_models"

pan_model = YOLO(str(MODELS_DIR / "pan_yolo.pt"))

face_net = cv2.dnn.readNetFromCaffe(str(FACE_MODELS_DIR / "deploy.prototxt"),
                                   str(FACE_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"))

pan_regex = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]{1}")

def extract_pan_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    extracted = {"Name": None, "Father_Name": None, "KYC_Number": None, "Photo": None}

    results = pan_model.predict(cv_image, conf=0.45)
    if results and len(results) > 0:
        boxes = results[0].boxes.data.tolist()
        for r in boxes:
            # r format: [x1,y1,x2,y2,conf,cls]
            x1, y1, x2, y2, conf, cls = r[:6]
            x1, y1, x2, y2 = map(int, map(round, (x1, y1, x2, y2)))
            crop = cv_image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_gray = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, crop_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try:
                import pytesseract
                text = pytesseract.image_to_string(crop_bin, lang='eng').strip()
            except:
                text = ""
            label = pan_model.names[int(cls)].lower()

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if label == "name":
                for line in lines:
                    if all(c.isalpha() or c.isspace() for c in line) and len(line.split()) >= 2:
                        extracted["Name"] = heuristic_name_split(line.title())
                        break
            elif label in ("father-s name", "father's name", "father name", "father"):
                for line in lines:
                    if all(c.isalpha() or c.isspace() for c in line) and len(line.split()) >= 2:
                        extracted["Father_Name"] = line.title()
                        break
            elif "pan" in label or "pan number" in label or "pan_no" in label:
                match = pan_regex.search(text.replace(" ", ""))
                if match:
                    extracted["KYC_Number"] = match.group()

    # face detection on right side regions as fallback
    h, w, _ = cv_image.shape
    regions_coords = [
        (int(0.1*h), int(0.5*h), int(0.7*w), int(0.95*w)),
        (int(0.5*h), int(0.9*h), int(0.7*w), int(0.95*w))
    ]
    for y1, y2, x1, x2 in regions_coords:
        region = cv_image[y1:y2, x1:x2]
        blob = cv2.dnn.blobFromImage(region, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        if detections.shape[2] > 0:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([x2-x1, y2-y1, x2-x1, y2-y1])
                    bx1, by1, bx2, by2 = box.astype("int")
                    face_crop = region[max(0, by1):min(y2-y1, by2), max(0, bx1):min(x2-x1, bx2)]
                    if face_crop.size == 0:
                        continue
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    buffered = io.BytesIO()
                    face_pil.save(buffered, format="JPEG")
                    extracted["Photo"] = base64.b64encode(buffered.getvalue()).decode()
                    break
            if extracted["Photo"]:
                break

    return extracted
