from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import fitz
import io
from pathlib import Path
import importlib

from ocr_utils.extract_aadhaar import extract_aadhaar_details_paddle
from ocr_utils.extract_pan import extract_pan_details
from ocr_utils.extract_voter import extract_voter_details
from ocr_utils.extract_passport import extract_passport_details
from ocr_utils.extract_dl import extract_driving_license_details
from ocr_utils.helpers import extract_photo, pil_bytes_to_jpeg_bytes

app = FastAPI(title="Offline KYC OCR Extractor API")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "static" / "frontend.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend UI"""
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/status")
async def status():
    """
    Check whether all required models and libraries are installed and working.
    Returns a JSON response with model availability info.
    """
    components = {
        "PaddleOCR": "paddleocr",
        "YOLOv8": "ultralytics",
        "OpenCV": "cv2",
        "PyMuPDF": "fitz",
        "PIL": "PIL",
    }

    status_report = {}
    for name, module_name in components.items():
        try:
            importlib.import_module(module_name)
            status_report[name] = "✅ Available"
        except Exception as e:
            status_report[name] = f"❌ Missing or failed to load: {e}"

    # Check custom extractors
    extractors = {
        "Aadhaar Extractor": extract_aadhaar_details_paddle,
        "PAN Extractor": extract_pan_details,
        "Voter Extractor": extract_voter_details,
        "Passport Extractor": extract_passport_details,
        "Driving License Extractor": extract_driving_license_details,
    }

    for name, func in extractors.items():
        try:
            if callable(func):
                status_report[name] = "✅ Loaded"
            else:
                status_report[name] = "⚠️ Not callable"
        except Exception as e:
            status_report[name] = f"❌ Failed to load: {e}"

    return JSONResponse(status_report)


@app.post("/extract")
async def extract_kyc(file: UploadFile, doc_type: str = Form(...)):
    """Extract details and return JSON inline"""
    try:
        content = await file.read()
        filename = file.filename.lower()

        # Convert PDF first page to image
        if filename.endswith(".pdf"):
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            pix = pdf_doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            content = buf.getvalue()
        else:
            content = pil_bytes_to_jpeg_bytes(content)

        doc_type = doc_type.lower()
        if doc_type == "aadhaar":
            details = extract_aadhaar_details_paddle(content)
        elif doc_type == "pan":
            details = extract_pan_details(content)
        elif doc_type == "voter":
            details = extract_voter_details(content)
        elif doc_type == "passport":
            details = extract_passport_details(content)
        elif doc_type == "driving_license":
            details = extract_driving_license_details(content)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported document type"})

        photo_b64 = extract_photo(content)
        return JSONResponse({"extracted_details": details, "photo_base64": photo_b64})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/extract/download_json")
async def download_json(file: UploadFile, doc_type: str = Form(...)):
    """Extract details and return JSON file for download"""
    try:
        content = await file.read()
        if file.filename.lower().endswith(".pdf"):
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            pix = pdf_doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            content = buf.getvalue()
        else:
            content = pil_bytes_to_jpeg_bytes(content)

        doc_type = doc_type.lower()
        if doc_type == "aadhaar":
            details = extract_aadhaar_details_paddle(content)
        elif doc_type == "pan":
            details = extract_pan_details(content)
        elif doc_type == "voter":
            details = extract_voter_details(content)
        elif doc_type == "passport":
            details = extract_passport_details(content)
        elif doc_type == "driving_license":
            details = extract_driving_license_details(content)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported document type"})

        photo_b64 = extract_photo(content)
        response = {"extracted_details": details, "photo_base64": photo_b64}
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
