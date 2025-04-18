from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import logging
import time
from websocket_server import WebSocketServer
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paddington_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Get the current directory for relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
logger.info(f"Static directory: {static_dir}")

# Ensure audio directory exists
audio_dir = os.path.join(static_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

app = FastAPI(title="Paddington Voice Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Get model paths from environment variables or use defaults
model_path = os.getenv("ZONOS_MODEL_PATH", "/data/models/zonos_hybrid/model.safetensors")
config_path = os.getenv("ZONOS_CONFIG_PATH", "/data/models/zonos_hybrid/config.json")
llm_model_path = os.getenv("LLM_MODEL_PATH", "/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf")
personality_path = os.getenv("PERSONALITY_PATH", 
                             os.path.join(os.path.dirname(current_dir), "data", "personality.json"))

# Log model paths
logger.info(f"Zonos Model Path: {model_path}")
logger.info(f"Zonos Config Path: {config_path}")
logger.info(f"LLM Model Path: {llm_model_path}")
logger.info(f"Personality Path: {personality_path}")

# Check if model files exist
if not os.path.exists(model_path):
    logger.warning(f"Zonos model not found at: {model_path}")
if not os.path.exists(config_path):
    logger.warning(f"Zonos config not found at: {config_path}")
if not os.path.exists(llm_model_path):
    logger.warning(f"LLM model not found at: {llm_model_path}")
if not os.path.exists(personality_path):
    logger.warning(f"Personality config not found at: {personality_path}")


from simulator_llama import Simulator
websocket_server = WebSocketServer(model_path, config_path)

class TextInput(BaseModel):
    text: str

class ChatHistoryInput(BaseModel):
    messages: List[Dict[str, str]]
    
class AudioResponse(BaseModel):
    text: str
    audio_url: Optional[str] = None
    error: Optional[str] = None

@app.post("/process-text")
async def process_text(input_data: TextInput):
    """Process text input directly, without audio"""
    try:
        # Generate a response using the simulator
        start_time = time.time()
        response = await websocket_server.simulator.process_text(input_data.text)
        logger.info(f"Text processing time: {time.time() - start_time:.2f}s")
        
        return response
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "text": "I encountered an error processing your message."}
        )

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Handle audio file upload and process it"""
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be an audio file"}
            )
            
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(current_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(uploads_dir, f"{file_id}.webm")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        logger.info(f"Saved audio file to {file_path}")
        
        # Process the audio file
        start_time = time.time()
        response = await websocket_server.process_audio_file(file_path)
        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
        
        # Clean up the uploaded file (consider removing if you need to debug)
        # os.remove(file_path)
        
        return response
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing audio: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "text": "I encountered an error processing your audio."}
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming
    """
    try:
        await websocket_server.handle_websocket(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)

@app.get("/health")
async def health_check():
    """
    Health check endpoint with detailed status
    """
    try:
        # Check if the TTS model is loaded
        tts_loaded = websocket_server.simulator.tts is not None
        llm_loaded = websocket_server.simulator.llm is not None
        
        return {
            "status": "healthy",
            "tts_model_loaded": tts_loaded,
            "llm_model_loaded": llm_loaded,
            "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve audio files from the static/audio directory
    This provides a direct path to access audio files at /audio/filename
    """
    audio_path = os.path.join(static_dir, "audio", filename)
    abs_path = os.path.abspath(audio_path)
    
    logger.info(f"Received request for audio file: {filename}")
    logger.info(f"Looking for file at: {abs_path}")
    
    if not os.path.exists(abs_path):
        logger.error(f"Audio file not found: {abs_path}")
        # Try with alternative extensions
        alt_extensions = ['.wav', '.webm', '.mp3']
        base_name = os.path.splitext(filename)[0]
        
        for ext in alt_extensions:
            alt_filename = base_name + ext
            alt_path = os.path.join(static_dir, "audio", alt_filename)
            if os.path.exists(alt_path):
                logger.info(f"Found alternative file: {alt_path}")
                abs_path = os.path.abspath(alt_path)
                break
        else:
            # No alternative found
            return JSONResponse(
                status_code=404,
                content={"error": f"Audio file not found: {filename}"}
            )
    
    # Determine the correct content type based on file extension
    file_ext = os.path.splitext(abs_path)[1].lower()
    if file_ext == '.wav':
        media_type = "audio/wav"
    elif file_ext == '.mp3':
        media_type = "audio/mpeg"
    elif file_ext == '.webm':
        media_type = "audio/webm"
    else:
        media_type = "application/octet-stream"
    
    file_size = os.path.getsize(abs_path)
    logger.info(f"Serving audio file: {abs_path}, size: {file_size} bytes, type: {media_type}")
    
    # If file size is zero, return error
    if file_size == 0:
        logger.error(f"Audio file is empty (0 bytes): {abs_path}")
        return JSONResponse(
            status_code=500,
            content={"error": "Audio file is empty or corrupted"}
        )
    
    return FileResponse(abs_path, media_type=media_type)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests and responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response

@app.on_event("startup")
async def startup_event():
    """Runs on application startup"""
    app.state.start_time = time.time()
    logger.info("Paddington API starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Runs on application shutdown"""
    logger.info("Paddington API shutting down")

if __name__ == "__main__":
    import uvicorn
    
    # Check if using SSL
    use_ssl = os.getenv("USE_SSL", "False").lower() == "true"
    
    if use_ssl:
        # SSL configuration
        ssl_cert = os.getenv("SSL_CERT", os.path.join(current_dir, "snakeoil.pem"))
        ssl_key = os.getenv("SSL_KEY", os.path.join(current_dir, "snakeoil.key"))
        
        if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
            logger.info("Starting server with SSL")
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=8000, 
                log_level="info",
                ssl_certfile=ssl_cert,
                ssl_keyfile=ssl_key,
                ws_ping_interval=20,
                ws_ping_timeout=30
            )
        else:
            logger.warning(f"SSL certificates not found at {ssl_cert} and {ssl_key}. Falling back to HTTP.")
            use_ssl = False
    
    if not use_ssl:
        logger.info("Starting server without SSL")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            ws_ping_interval=20,
            ws_ping_timeout=30
        )
