from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from main import process_image, process_pdf, process_image_group
from src.pdf_service import PDFService

app = FastAPI(title="Document AI API")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

pdf_service = PDFService(TEMP_DIR)


# ==========================
# HELPER FUNCTION
# ==========================
def save_file(upload_file: UploadFile):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{file_id}_{upload_file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path


# ==========================
# IMAGE API
# ==========================
@app.post("/process-image")
async def process_image_api(file: UploadFile = File(...)):
    try:
        path = save_file(file)

        result = process_image(path)

        return JSONResponse(content={
            "status": "success",
            "data": result
        })

    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)


# ==========================
# PDF API
# ==========================
@app.post("/process-pdf")
async def process_pdf_api(file: UploadFile = File(...)):
    try:
        path = save_file(file)

        pdf_name, result = process_pdf(path, pdf_service)

        return JSONResponse(content={
            "status": "success",
            "document": pdf_name,
            "data": result
        })

    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)


# ==========================
# MULTIPLE FILES (FOLDER)
# ==========================
@app.post("/process-multiple")
async def process_multiple_api(files: list[UploadFile] = File(...)):
    try:
        paths = []

        for file in files:
            path = save_file(file)
            paths.append(path)

        result = process_image_group(paths)

        return JSONResponse(content={
            "status": "success",
            "data": result
        })

    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)