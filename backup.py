from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import fitz  # PyMuPDF
import re
import base64
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import json
import tempfile

# Hardcoded tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Offline KYC OCR Extractor API")

# ------------------ OCR Preprocessing ------------------ #
def preprocess_image(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")  # PIL grayscale
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    img_cv = np.array(gray)
    return img_cv  # Always returns NumPy array

def ocr_image_bytes(image_bytes: bytes) -> str:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_image = preprocess_image(pil_image)  # returns a NumPy array
    text = pytesseract.image_to_string(processed_image, lang='eng')
    return text


def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Perform OCR on PDF bytes
    """
    text = ""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf_doc:
        # Render page to high-res image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("ppm")
        text += ocr_image_bytes(img_bytes) + "\n"
    return text

# ------------------ Photo Extraction ------------------ #
def extract_photo(image_bytes: bytes):
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        (h, w) = cv_image.shape[:2]

        # Load DNN face detector
        prototxt_path = r"C:\Users\ASUS\Desktop\kyc_ocr_app\face_models\deploy.prototxt"
        model_path = r"C:\Users\ASUS\Desktop\kyc_ocr_app\face_models\res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        # Prepare blob and detect faces
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        if detections.shape[2] == 0:
            return None
        # Take the first detected face
        confidence_threshold = 0.5
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face_img = cv_image[startY:endY, startX:endX]
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                buffered = io.BytesIO()
                face_pil.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode()
        return None
    except Exception as e:
        print("extract_photo error:", e)
        return None


# ------------------ Name Heuristic Split ------------------ #
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 :]', ' ', text)
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

# ------------------ Aadhaar Extraction ------------------ #
aadhaar_ocr = PaddleOCR(use_gpu=False, lang='en', use_angle_cls=True,
                        enable_mkldnn=False, use_textline_orientation=True,
                        rec_algorithm='SVTR_LCNet', det_algorithm='DB', ocr_version='PP-OCRv4')

aadhaar_model = YOLO("models/Aadhaar_Card.pt")

def preprocess_aadhaar_image(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
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
                extracted[cls] = text
    return extracted

def extract_aadhaar_details_paddle(image_bytes: bytes):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_cv = preprocess_aadhaar_image(pil_image)
    results = aadhaar_model(img_cv)
    detections = []
    if results and len(results) > 0:
        for r in results:
            for box, cls_idx in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                cls_name = aadhaar_model.names[int(cls_idx)]
                detections.append((cls_name, box))
    return extract_aadhaar_fields(img_cv, detections)

# ------------------ PAN Extraction (YOLO + Face) ------------------ #
from ultralytics import YOLO

# Load YOLO model once
yolo_model_path = "models/pan_yolo.pt"
pan_model = YOLO(yolo_model_path)

# Face detection DNN models
face_proto = r"face_models/deploy.prototxt"
face_model = r"face_models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

pan_regex = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]{1}")

def extract_pan_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    extracted = {"Name": None, "Father_Name": None, "KYC_Number": None, "Photo": None}

    # ------------------ YOLO Detection ------------------ #
    results = pan_model.predict(cv_image, conf=0.5)
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = map(int, r[:6])
        crop = cv_image[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, crop_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(crop_bin, lang='eng').strip()
        label = pan_model.names[cls].lower()

        if label == "name":
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines:
                if all(c.isalpha() or c.isspace() for c in line) and len(line.split()) >= 2:
                    extracted["Name"] = line.title()
                    break
        elif label == "father-s name":
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines:
                if all(c.isalpha() or c.isspace() for c in line) and len(line.split()) >= 2:
                    extracted["Father_Name"] = line.title()
                    break
        elif label == "pan number":
            match = pan_regex.search(text)
            if match:
                extracted["KYC_Number"] = match.group()

    # ------------------ Face Detection (Upper/Lower Right) ------------------ #
    h, w, _ = cv_image.shape
    regions_coords = [
        (int(0.1*h), int(0.5*h), int(0.7*w), int(0.95*w)),  # upper-right
        (int(0.5*h), int(0.9*h), int(0.7*w), int(0.95*w))   # lower-right
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
                    face_crop = region[by1:by2, bx1:bx2]
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    buffered = io.BytesIO()
                    face_pil.save(buffered, format="JPEG")
                    extracted["Photo"] = base64.b64encode(buffered.getvalue()).decode()
                    break
            if extracted["Photo"]:
                break

    return extracted


# ------------------ Voter ID Extraction ------------------ #
VOTER_CLASSES = [
    "Address", "Age", "DOB", "Card Voter ID 1 Back", "Card Voter ID 2 Front",
    "Card Voter ID 2 Back", "Card Voter ID 1 Front", "DOB", "Date of Issue",
    "Election", "Father", "Gender", "Name", "Point", "Portrait", "Symbol",
    "Voter ID"
]
REMOVE_FIELDS = ["Card Voter ID 1 Front", "Election", "Symbol"]

voter_model_path = r"models\voter_id.pt"
voter_model = YOLO(voter_model_path)

FACE_PROTO = r"face_models\deploy.prototxt"
FACE_MODEL = r"face_models\res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

ocr = PaddleOCR(
    use_gpu=False,
    lang='en',
    use_angle_cls=True,
    enable_mkldnn=False,
    use_textline_orientation=True,
    rec_algorithm='SVTR_LCNet',
    det_algorithm='DB',
    ocr_version='PP-OCRv4'
)

voter_regex = re.compile(r'[A-Z]{3}[0-9]{7}')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    gray = cv2.resize(gray, (w*2, h*2))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def preprocess_voter_id_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = threshed.shape
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
        return ""

def backup_tesseract(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config="--psm 6").strip()

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 :]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def heuristic_name_split(full_name):
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
            _, buffer = cv2.imencode('.jpg', face_crop)
            return base64.b64encode(buffer).decode('utf-8')
    return "Face not detected"

def extract_voter_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_image.shape

    results = voter_model(cv_image)
    extracted = {"Name": None, "Father": None, "Portrait": None}
    detected_data = {}

    # First pass: extract all fields except Portrait
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
                text = backup_tesseract(roi_proc)

            if "voterid" in class_name.lower():
                match = voter_regex.search(text.replace(" ", ""))
                text = match.group(0) if match else text

            if class_name.lower() == "name":
                detected_data["Name"] = heuristic_name_split(text)
            elif class_name.lower() == "father":
                detected_data["Father"] = text.title()
            else:
                detected_data[class_name] = text if text else "N/A"

    # Second pass: handle Portrait
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

    # Ensure JSON keys
    for key in ["Name", "Father", "Portrait"]:
        if key not in detected_data:
            detected_data[key] = None

    return detected_data

# ------------------ Passport extraction ------------------ #
ocr = PaddleOCR(use_gpu=False, lang='en', use_angle_cls=True)
face_proto = r"face_models/deploy.prototxt"
face_model = r"face_models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
passport_model_path = r"models\passport.pt"
passport_model = YOLO(passport_model_path)

passport_classes = [
    "Address", "Code", "DOB", "DOI", "EXP", "Gender",
    "MRZ2", "MRZ1", "MRZ2", "Name", "Nationality", "Nation", "POI"
]

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
        return pytesseract.image_to_string(roi, config="--psm 6").strip()

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9/ :]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def heuristic_name_split(full_name):
    parts = full_name.split()
    if len(parts) >= 3:
        return {"first_name": parts[0], "middle_name": " ".join(parts[1:-1]), "last_name": parts[-1]}
    elif len(parts) == 2:
        return {"first_name": parts[0], "middle_name": "", "last_name": parts[1]}
    elif len(parts) == 1:
        return {"first_name": parts[0], "middle_name": "", "last_name": ""}
    else:
        return {"first_name": "", "middle_name": "", "last_name": ""}

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

def extract_passport_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = passport_model(cv_img)

    data = {
        "Name": None, "DOB": None, "DOI": None, "EXP": None,
        "Gender": None, "Nationality": None, "Code": None, "Portrait": None
    }

    name_detections = []  # collect all name ROIs and text

    # --- First pass: extract text for all detected boxes ---
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = passport_classes[cls_id] if cls_id < len(passport_classes) else "Unknown"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = cv_img[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            text = safe_ocr(roi)
            text = clean_text(text)

            if label == "Name" and text:
                name_detections.append((y1, text))  # store y for sorting later
            elif label in ["DOB", "DOI", "EXP", "Gender", "Nationality", "Code"]:
                data[label] = text

    # --- Sort name detections top-to-bottom (passport layout consistency) ---
    name_detections.sort(key=lambda x: x[0])

    # --- Combine all detected names ---
    if len(name_detections) > 0:
        all_names = [nd[1] for nd in name_detections]
        combined_name = " ".join(all_names).strip()

        # Assign first detected as surname, next as first name
        if len(all_names) == 1:
            data["Name"] = heuristic_name_split(all_names[0])
        elif len(all_names) >= 2:
            data["Name"] = {
                "first_name": all_names[1],
                "middle_name": " ".join(all_names[2:-1]) if len(all_names) > 2 else "",
                "last_name": all_names[0]
            }
        else:
            data["Name"] = heuristic_name_split(combined_name)

    # --- Detect portrait last ---
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = passport_classes[cls_id]
            if label.lower() == "poi":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = cv_img[y1:y2, x1:x2]
                data["Portrait"] = detect_face(roi)

    return data

# ------------------ Driving License Extraction ------------------ #

dl_model_path = r"models/driving_licence.pt"
dl_model = YOLO(dl_model_path)
dl_classes = [
    "Address",
    "Blood Group",
    "DL No",
    "DOB",
    "Name",
    "Relation With",
    "RTO",
    "State",
    "Vehicle Type"
]

def extract_driving_license_details(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = dl_model(cv_img)
    data = {
        "DL No": None, "Name": None, "DOB": None, "Relation With": None,
        "Address": None, "RTO": None, "State": None, "Vehicle Type": None,
        "Blood Group": None, "Portrait": None
    }

    name_detections = []

    # --- Pass 1: OCR extraction ---
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
                data[label] = text

    # --- Handle Multiple Name Detections (Top = Surname, Below = First Name) ---
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

    # --- Portrait detection (optional if present) ---
    # Some DL models might not have a "POI" label, so we’ll detect face from full image
    data["Portrait"] = detect_face(cv_img)

    return data

# ------------------ Frontend HTML (ChatGPT-style) ------------------ #
def render_frontend():
    return """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f7f8fa;
            margin: 0; padding: 0;
        }

        .container {
            max-width: 900px; margin: 40px auto;
            background: #fff; padding: 30px; border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }

        h2 { color: #333; margin-bottom: 20px; }

        label { font-weight: bold; margin-top: 10px; display: block; }
        select, input[type=file], button {
            margin-top: 8px; margin-bottom: 15px;
            width: 100%; padding: 12px; font-size: 16px;
            border-radius: 10px; border: 1px solid #ccc;
        }

        button {
            background-color: #4a90e2; color: #fff; cursor: pointer;
            transition: 0.2s; font-weight: bold;
        }

        button:hover { background-color: #357ABD; }

        #chat-container {
            display: flex; flex-direction: column; gap: 15px;
            margin-top: 25px;
        }

        .chat-bubble {
            max-width: 90%; padding: 15px 20px;
            border-radius: 20px; line-height: 1.5; word-wrap: break-word;
        }

        .user { align-self: flex-end; background-color: #4a90e2; color: #fff; }
        .bot { align-self: flex-start; background-color: #f1f0f0; color: #333; }

        .bot img {
            display: block; margin-top: 10px;
            max-width: 200px; border-radius: 10px; border: 1px solid #ccc;
        }

        .json-block {
            background: #e8e8e8; padding: 10px; border-radius: 10px;
            font-family: monospace; white-space: pre-wrap; overflow-x: auto;
            margin-top: 10px;
        }
    </style>

    <div class="container">
        <h2>Offline KYC OCR Extractor</h2>
        <form id="extract-form" enctype="multipart/form-data">
            <label for="doc_type">Document Type:</label>
            <select name="doc_type" id="doc_type">
                <option value="pan">PAN</option>
                <option value="aadhaar">Aadhaar</option>
                <option value="voter">Voter ID</option>
                <option value="passport">Passport</option>
                <option value="driving_license">Driving License</option>
            </select>

            <label for="file">Upload Document:</label>
            <input type="file" id="file" name="file" accept=".jpg,.jpeg,.png,.pdf">

            <button type="button" onclick="uploadExtract()">Extract & View JSON</button>
            <button type="button" onclick="uploadDownload()">Extract & Download JSON</button>
        </form>

        <div id="chat-container"></div>
    </div>

    <script>
        async function uploadExtract() {
            const form = document.getElementById('extract-form');
            const data = new FormData(form);
            addUserBubble("Uploading and extracting data...");

            try {
                const res = await fetch('/extract', { method: 'POST', body: data });
                const json = await res.json();
                addBotBubble(json);
            } catch (err) {
                addBotBubble({ error: err.toString() });
            }
        }

        async function uploadDownload() {
            const form = document.getElementById('extract-form');
            const data = new FormData(form);
            const docType = document.getElementById('doc_type').value;

            try {
                const res = await fetch('/extract/download_json', { method: 'POST', body: data });
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = docType + "_data.json";
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            } catch (err) {
                addBotBubble({ error: err.toString() });
            }
        }

        function addUserBubble(text) {
            const chat = document.getElementById('chat-container');
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble user';
            bubble.textContent = text;
            chat.appendChild(bubble);
            bubble.scrollIntoView({ behavior: 'smooth' });
        }

        function addBotBubble(json) {
            const chat = document.getElementById('chat-container');
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble bot';

            if(json.error) {
                bubble.textContent = "Error: " + json.error;
            } else {
                let htmlContent = "";

                // Extracted details as collapsible JSON block
                htmlContent += "<strong>Extracted Details:</strong>";
                htmlContent += "<div class='json-block'>" + JSON.stringify(json.extracted_details, null, 4) + "</div>";

                // Display extracted photo if available
                if(json.photo_base64){
                    htmlContent += "<strong>Extracted Photo:</strong><br><img src='data:image/jpeg;base64," + json.photo_base64 + "' />";
                }

                bubble.innerHTML = htmlContent;
            }

            chat.appendChild(bubble);
            bubble.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
    """

# FastAPI route
@app.get("/", response_class=HTMLResponse)
async def index():
    return render_frontend()

@app.post("/extract")
async def extract_kyc(file: UploadFile, doc_type: str = Form(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            pix = pdf_doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        if doc_type.lower() == "aadhaar":
            details = extract_aadhaar_details_paddle(img_bytes)
        elif doc_type.lower() == "pan":
            details = extract_pan_details(img_bytes)
        elif doc_type.lower() == "voter":
            details = extract_voter_details(img_bytes)
        elif doc_type.lower() == "passport":
            details = extract_passport_details(img_bytes)
        elif doc_type.lower() == "driving_license":
            details = extract_driving_license_details(img_bytes)
        photo_b64 = extract_photo(img_bytes)
        return JSONResponse({"extracted_details": details, "photo_base64": photo_b64})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/extract/download_json")
async def download_json(file: UploadFile, doc_type: str = Form(...)):
    """
    Extract KYC information from uploaded file and return JSON for download.
    Supports images (jpg, png) and PDFs.
    """
    try:
        contents = await file.read()
        # Convert PDF first page to image
        if file.filename.lower().endswith(".pdf"):
            pdf_doc = fitz.open(stream=contents, filetype="pdf")
            pix = pdf_doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            contents = buf.getvalue()

        # Extract details based on document type
        doc_type = doc_type.lower()
        if doc_type == "aadhaar":
            extracted = extract_aadhaar_details_paddle(contents)
        elif doc_type == "pan":
            extracted = extract_pan_details(contents)
        elif doc_type == "voter":
            extracted = extract_voter_details(contents)
        elif doc_type == "passport":
            extracted = extract_passport_details(contents)
        elif doc_type == "driving_license":
            extracted = extract_driving_license_details(contents)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported doc type"})

        # Extract face photo (optional)
        photo_b64 = extract_photo(contents)

        response = {"extracted_details": extracted, "photo_base64": photo_b64}
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})