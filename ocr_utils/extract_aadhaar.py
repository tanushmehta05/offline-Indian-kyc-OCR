# ocr_utils/extract_aadhaar.py
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import numpy as np
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from .helpers import heuristic_name_split, clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

aadhaar_ocr = PaddleOCR(use_gpu=False, lang='en', use_angle_cls=True,
                        enable_mkldnn=False, use_textline_orientation=True,
                        rec_algorithm='SVTR_LCNet', det_algorithm='DB', ocr_version='PP-OCRv4')

_aadhaar_model = None
def _get_aadhaar_model():
    global _aadhaar_model
    if _aadhaar_model is None:
        _aadhaar_model = YOLO(str(MODELS_DIR / "Aadhaar_Card.pt"))
    return _aadhaar_model

def _preprocess(img_pil: Image.Image):
    gray = img_pil.convert("L")
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    img_cv = np.array(gray)
    h, w = img_cv.shape
    if h < 800:
        img_cv = cv2.resize(img_cv, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    return img_cv

def extract_aadhaar_fields(image: np.ndarray, detections: list):
    extracted = {}
    for cls, bbox in detections:
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if cls.lower() == "photo":
            _, buffer = cv2.imencode('.jpg', crop)
            extracted["Photo"] = base64.b64encode(buffer).decode('utf-8')
        else:
            ocr_result = aadhaar_ocr.ocr(crop)
            text = " ".join([line[1][0] for line in ocr_result[0]]) if ocr_result else ""
            text = text.strip()
            if cls.lower() == "name":
                extracted[cls] = heuristic_name_split(text)
            else:
                extracted[cls] = clean_text(text)
    return extracted

def extract_aadhaar_details_paddle(image_bytes: bytes):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_cv = _preprocess(pil_image)
    model = _get_aadhaar_model()
    results = model(img_cv)
    detections = []
    if results and len(results) > 0:
        for r in results:
            for box, cls_idx in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                cls_name = model.names[int(cls_idx)]
                detections.append((cls_name, box))
    return extract_aadhaar_fields(img_cv, detections)
