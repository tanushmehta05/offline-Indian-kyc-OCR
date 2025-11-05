**Offline KYC OCR Extractor**

🔍 Intelligent, Offline Identity Document Extraction using AI and OCR
🚀 Overview

Offline KYC OCR Extractor is a complete AI-powered document processing system built using FastAPI, YOLOv8, and PaddleOCR.
It enables automated extraction of key fields (Name, DOB, ID Number, Photo, etc.) from Indian identity documents — completely offline (no API calls or cloud dependencies).

Supported document types:

🪪 Aadhaar Card

💳 PAN Card

🧾 Voter ID

🛂 Passport

🚗 Driving License

⚙️ Tech Stack
Category	Technologies / Libraries Used
Backend Framework	FastAPI
OCR Engine	PaddleOCR, Tesseract
Object Detection	YOLOv8 (Ultralytics)
Image Processing	OpenCV, Pillow (PIL)
PDF Handling	PyMuPDF (fitz)
Face Detection	OpenCV DNN (Caffe Model)
Frontend	HTML + JavaScript (ChatGPT-style UI)
Environment	100% Offline (no API usage)
🧩 Folder Structure
kyc_ocr_app/
│
├── main.py                        # FastAPI app entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
│
├── models/                         # YOLOv8 trained models
│   ├── Aadhaar_Card.pt
│   ├── pan_yolo.pt
│   ├── voter_id.pt
│   ├── passport.pt
│   └── driving_licence.pt
│
├── face_models/                    # Face detection models
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── ocr_utils/                      # Modular extraction scripts
│   ├── __init__.py
│   ├── helpers.py
│   ├── extract_aadhaar.py
│   ├── extract_pan.py
│   ├── extract_voter.py
│   ├── extract_passport.py
│   └── extract_dl.py
│
└── static/
    └── frontend.html               # ChatGPT-style upload & extraction UI

🧠 How It Works

📤 Upload Document
User uploads an image or PDF file of any supported KYC document.

🎯 YOLO Detection
YOLOv8 detects predefined fields such as Name, DOB, ID Number, and Photo regions.

🔠 OCR Extraction
PaddleOCR and Tesseract extract text from each detected region accurately.

🧍 Face Extraction
OpenCV’s DNN detects and extracts the document portrait as a base64 image.

📦 JSON Output
The system returns a structured JSON containing all extracted fields and encoded photo.

🧪 Example Output
{
  "extracted_details": {
    "Name": {
      "first_name": "John",
      "middle_name": "",
      "last_name": "Doe"
    },
    "DOB": "05/06/1995",
    "Aadhaar_Number": "1234 5678 9012"
  },
  "photo_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}

🖥️ User Interface

A clean, minimal ChatGPT-style frontend built using pure HTML + JavaScript.

💡 Features

Upload any supported ID card (JPG, PNG, PDF)

Choose document type (Aadhaar, PAN, etc.)

View results as formatted JSON

Download extracted JSON with a single click

View extracted photo (if available)

🧰 Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/kyc_ocr_app.git
cd kyc_ocr_app

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate      # for Linux/Mac
venv\Scripts\activate         # for Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the FastAPI Server
uvicorn main:app --reload

5️⃣ Open in Browser

👉 http://127.0.0.1:8000

📦 Requirements

Your requirements.txt should include:

fastapi
uvicorn
pillow
pytesseract
paddleocr
ultralytics
opencv-python
numpy
PyMuPDF

🧠 Architecture Overview
flowchart TD
    A[Upload Document] --> B[YOLOv8 Detection]
    B --> C[PaddleOCR + Tesseract Extraction]
    C --> D[OpenCV Face Detection]
    D --> E[Data Structuring + Cleaning]
    E --> F[JSON Output + Photo Encoding]
    F --> G[Frontend Display / Download]

✨ Key Features

✅ Fully Offline (no API calls required)
✅ Multi-document ID support (Aadhaar, PAN, Voter ID, Passport, DL)
✅ AI-based text region detection (YOLOv8)
✅ Dual OCR (PaddleOCR + Tesseract fallback)
✅ Automatic face extraction (base64 encoded)
✅ JSON export + Chat-style visualization
✅ Modular design for easy extension

🧩 Core Modules Overview
Module	Purpose
extract_aadhaar.py	Extract Aadhaar name, DOB, number, and photo using YOLO + PaddleOCR
extract_pan.py	Extract PAN name, father’s name, and PAN number
extract_voter.py	Extract voter name, father, DOB, and photo
extract_passport.py	Extract passport name, nationality, gender, and expiry details
extract_dl.py	Extract driving license number, name, DOB, and blood group
helpers.py	Shared preprocessing and face extraction functions

📸 Sample Frontend Preview

-------------------------------------------------------
| Offline KYC OCR Extractor                          |
-------------------------------------------------------
| [ Select Document Type ▼ ]                         |
| [ Upload File (JPG/PNG/PDF) ]                      |
| [ Extract & View JSON ] [ Extract & Download JSON ] |
-------------------------------------------------------
| Chat-style Output:                                 |
| { Extracted JSON ... }                             |
| [ Extracted Photo 🧍 ]                             |
-------------------------------------------------------

💡 Future Enhancements

Add signature detection & verification

Support for regional language OCR

Integration with document authenticity detection

Batch upload & queue processing

Optional API key-based access control


Developed by Tanush


🏁 License

This project is released under the MIT License — free to use, modify, and distribute with proper attribution.