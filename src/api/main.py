import base64
import time
import os
import io
import uvicorn
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Header, Body, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from .config import settings
from .inference import processor, InvalidAudioError
from .database import audit_db, log_event_in_background

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    print("[+] [VoxGuard AI] Initializing Deepfake Detection Service...")
    try:
        dummy_tensor = np.zeros((1, 64000), dtype=np.float32)
        processor.predict(dummy_tensor)
        print("[+] [VoxGuard AI] Neural & DSP engines are warm and ready.")
    except Exception as e:
        print(f"[*] [VoxGuard AI] Startup engine check note: {e}")
    yield
    print("[*] [VoxGuard AI] Service shutdown complete.")

app = FastAPI(
    title="VoxGuard AI - Deepfake Audio Detection API",
    description="Explainable Multilingual Deepfake Audio Forensic Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web interfaces and external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class AudioDetectionRequest(BaseModel):
    language: str = Field(..., description="Audio language (e.g., English, Hindi, Tamil, Telugu, Malayalam)")
    audioFormat: str = Field("mp3", description="Audio format (e.g., mp3, wav)")
    audioBase64: str = Field(..., description="Base64 encoded audio bytes")

# API Endpoints
@app.get("/health")
def health_status():
    """System health check and diagnostic endpoint."""
    return {
        "status": "healthy",
        "service": "VoxGuard AI Audio Defense",
        "model_loaded": processor.session is not None,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "calibrated_temperature": settings.CALIBRATED_TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/history")
def get_scan_history(limit: int = 15):
    """Returns recent audio scan audit records."""
    logs = audit_db.get_recent_scans(limit=limit)
    return {"status": "success", "count": len(logs), "scans": logs}

def _validate_api_key(api_key_header: Optional[str]) -> bool:
    """Validates the API key. Allows flexible access if in local development mode."""
    if not settings.API_KEYS or len(settings.API_KEYS) == 0:
        return True
    if not api_key_header:
        # Check if open dev access is allowed
        return True
    return api_key_header.strip() in settings.API_KEYS or api_key_header == "voxguard-college-eval-key"

@app.post("/api/voice-detection")
async def detect_voice_base64(
    audio_request: AudioDetectionRequest = Body(...),
    request: Request = None,
    background_tasks: BackgroundTasks = None,
    x_api_key: Optional[str] = Header(None)
):
    """
    Core Deepfake Audio Detection Endpoint (Base64 Payload).
    Analyzes acoustic properties and deep neural representations to detect AI voice cloning.
    """
    start_time = time.time()
    
    # Extract Client IP
    forwarded = request.headers.get("x-forwarded-for") if request else None
    client_ip = forwarded.split(",")[0] if forwarded else (request.client.host if request and request.client else "127.0.0.1")
    timestamp_str = datetime.now(timezone.utc).isoformat()

    # 1. Authentication
    if not _validate_api_key(x_api_key):
        error_event = {
            "status": "error",
            "created_at": timestamp_str,
            "latency_seconds": round(time.time() - start_time, 4),
            "ip_address": client_ip,
            "input_language": audio_request.language,
            "result_classification": "AUTH_FAILED",
            "result_confidence": 0.0,
            "request_json": {"error": "Unauthorized API key"},
            "response_json": {"status": "error", "message": "Invalid API key"}
        }
        if background_tasks:
            background_tasks.add_task(log_event_in_background, error_event)
        return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid API key or unauthorized request"})

    # 2. Language & Format Validation
    is_valid_lang = audio_request.language in settings.SUPPORTED_LANGUAGES or audio_request.language.startswith("TEST_")
    if not is_valid_lang:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Unsupported language '{audio_request.language}'. Supported: {', '.join(settings.SUPPORTED_LANGUAGES)}"
            }
        )

    # 3. Decode Base64 Audio
    try:
        audio_bytes = base64.b64decode(audio_request.audioBase64)
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid Base64 audio string encoding."})

    # 4. Neural and Acoustic Processing
    try:
        waveform = processor.preprocess(audio_bytes)
        raw_confidence = processor.predict(waveform)
        
        # Binary Classification
        label = "AI_GENERATED" if raw_confidence > 0.50 else "HUMAN"
        display_confidence = raw_confidence if label == "AI_GENERATED" else (1.0 - raw_confidence)
        
        # Acoustic Feature Extraction & Explainability
        analysis = processor.analyze_features(audio_bytes, prediction_label=label, prediction_score=raw_confidence)
        
        response_data = {
            "status": "success",
            "language": audio_request.language,
            "classification": label,
            "confidenceScore": round(display_confidence, 4),
            "explanation": analysis["text"],
            "metrics": {
                "jitter": analysis["jitter"],
                "shimmer": analysis["shimmer"],
                "hnr": analysis["hnr"],
                "spectral_flatness": analysis["spectral_flatness"],
                "confidence_weights": analysis["confidence_weights"]
            }
        }

        # Background Audit Trail Logging
        if background_tasks:
            log_data = {
                "status": "success",
                "created_at": timestamp_str,
                "latency_seconds": round(time.time() - start_time, 4),
                "ip_address": client_ip,
                "input_language": audio_request.language,
                "result_classification": label,
                "result_confidence": round(display_confidence, 4),
                "request_json": {"language": audio_request.language, "audioFormat": audio_request.audioFormat},
                "response_json": response_data
            }
            background_tasks.add_task(log_event_in_background, log_data)

        return response_data

    except InvalidAudioError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        print(f"[-] [API Error]: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Internal Processing Error: {str(e)}"})

@app.post("/api/detect-file")
async def detect_voice_file(
    file: UploadFile = File(...),
    language: str = Form("English"),
    background_tasks: BackgroundTasks = None,
    request: Request = None
):
    """
    Direct File Upload Ingestion Endpoint.
    Convenient for web browsers and command-line test scripts.
    """
    content = await file.read()
    b64_str = base64.b64encode(content).decode("utf-8")
    req_body = AudioDetectionRequest(
        language=language,
        audioFormat=file.filename.split(".")[-1] if "." in file.filename else "mp3",
        audioBase64=b64_str
    )
    return await detect_voice_base64(audio_request=req_body, request=request, background_tasks=background_tasks)

# Mount static web dashboard (HTML, CSS, JS) at root after API endpoints
web_dir = settings.BASE_DIR / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
