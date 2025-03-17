from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
from model_integration import PersonalitySimulator
from websocket_server import stream_manager
import logging
import uuid
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personality Simulator API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the simulator with default models
logger.info("Initializing PersonalitySimulator...")
logger.info(f"LLaMA model path: /data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf")
logger.info(f"Voice model path: /data/models/voice/voice_model.pth")
logger.info(f"Personality data path: /home/murtaza/paddington/data/personality.json")

simulator = PersonalitySimulator(
    llama_model_path="/data/models/llama/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
    voice_model_path="/data/models/voice/voice_model.pth",
    personality_data_path="/home/murtaza/paddington/data/personality.json"
)
logger.info("PersonalitySimulator initialized successfully")

class TextInput(BaseModel):
    text: str

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming
    """
    client_id = str(uuid.uuid4())
    try:
        await stream_manager.connect(websocket, client_id)
        
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()
            
            # Process audio and generate response
            await stream_manager.process_audio(client_id, data)
            
    except WebSocketDisconnect:
        await stream_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await stream_manager.disconnect(client_id)

@app.post("/process-text")
async def process_text(input_data: TextInput):
    """
    Process text input and return generated response with audio
    """
    try:
        logger.info(f"Received text input: {input_data.text}")
        response = simulator.process_interaction(audio_input=False, input_text=input_data.text)
        logger.info(f"Generated response: {response}")
        return {
            "response_text": response["response_text"],
            "audio_path": response["audio_path"]
        }
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """
    Process audio input and return generated response with audio
    """
    try:
        # Save uploaded audio to the audio directory
        audio_filename = f"upload_{audio_file.filename}"
        audio_path = os.path.join(simulator.audio_dir, audio_filename)
        
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Process audio
        response = simulator.process_interaction(audio_input=True)
        
        # Cleanup uploaded file
        try:
            os.remove(audio_path)
        except:
            pass
        
        return {
            "transcribed_text": response["input_text"],
            "response_text": response["response_text"],
            "audio_path": response["audio_path"]
        }
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-response-audio/{filename}")
async def get_response_audio(filename: str):
    """
    Get the generated audio response
    """
    try:
        file_path = os.path.join(simulator.audio_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error retrieving audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 