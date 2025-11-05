# ocr_utils/helpers.py
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import cv2
import numpy as np
import re

BASE_DIR = Path(__file__).resolve().parent.parent
FACE_MODELS_DIR = BASE_DIR / "face_models"

# load face DNN
_face_net = None
def _get_face_net():
    global _face_net
    if _face_net is None:
        proto = str(FACE_MODELS_DIR / "deploy.prototxt")
        model = str(FACE_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")
        _face_net = cv2.dnn.readNetFromCaffe(proto, model)
    return _face_net

def preprocess_image_pil(pil_image: Image.Image) -> np.ndarray:
    gray = pil_image.convert("L")
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    img_cv = np.array(gray)
    return img_cv

def pil_bytes_to_jpeg_bytes(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def extract_photo(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        (h, w) = cv_image.shape[:2]
        net = _get_face_net()
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        confidence_threshold = 0.5
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                face_img = cv_image[startY:endY, startX:endX]
                if face_img.size == 0:
                    continue
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                buffered = io.BytesIO()
                face_pil.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode()
        return None
    except Exception as e:
        return None

def clean_text(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9 :/\\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def heuristic_name_split(full_name: str):
    full_name = clean_text(full_name)
    parts = full_name.split()
    if len(parts) >= 3:
        return {"first_name": parts[0], "middle_name": " ".join(parts[1:-1]), "last_name": parts[-1]}
    elif len(parts) == 2:
        return {"first_name": parts[0], "middle_name": "", "last_name": parts[1]}
    elif len(parts) == 1:
        return {"first_name": parts[0], "middle_name": "", "last_name": ""}
    else:
        return {"first_name": "", "middle_name": "", "last_name": ""}
