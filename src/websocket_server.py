from fastapi import WebSocket, WebSocketDisconnect
import logging
import os
import asyncio
import json
import time
import traceback
from typing import Dict, Any, Set, Optional
from simulator_llama import Simulator

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("websocket_server.log")
    ]
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_times: Dict[str, float] = {}
        
    async def connect(self, websocket: WebSocket) -> str:
        """Accept connection and return client ID"""
        await websocket.accept()
        client_id = str(id(websocket))
        self.active_connections[client_id] = websocket
        self.connection_times[client_id] = time.time()
        logger.info(f"Client connected: {client_id}")
        return client_id
        
    def disconnect(self, client_id: str):
        """Remove connection from active connections"""
        if client_id in self.active_connections:
            connection_duration = time.time() - self.connection_times.get(client_id, time.time())
            logger.info(f"Client disconnected: {client_id}, session duration: {connection_duration:.2f}s")
            del self.active_connections[client_id]
            if client_id in self.connection_times:
                del self.connection_times[client_id]
    
    async def send_json(self, client_id: str, message: Dict[str, Any]):
        """Send JSON to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {str(e)}")
                return False
        return False
    
    async def broadcast_json(self, message: Dict[str, Any]):
        """Send JSON to all connected clients"""
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

class WebSocketServer:
    def __init__(self, model_path: str, config_path: str):
        """Initialize the WebSocket server"""
        try:
            self.simulator = Simulator(model_path, config_path)
            self.manager = ConnectionManager()
            self.processing_clients: Set[str] = set()
            
            # Ensure temp directory exists
            self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(self.temp_dir, exist_ok=True)
            
            logger.info(f"WebSocketServer initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocketServer: {str(e)}", exc_info=True)
            raise
        
    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections"""
        client_id = await self.manager.connect(websocket)
        
        # Send initial connection message
        try:
            await websocket.send_json({
                "type": "connection_status",
                "connected": True,
                "message": "Connected to Paddington voice assistant"
            })
        except Exception as e:
            logger.error(f"Failed to send welcome message: {str(e)}")
        
        try:
            while True:
                # Check if client is already being processed
                if client_id in self.processing_clients:
                    logger.warning(f"Client {client_id} already has a request being processed. Ignoring new request.")
                    await asyncio.sleep(0.5)
                    continue
                
                try:
                    # Receive audio data with a timeout
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                    
                    # Add client to processing set
                    self.processing_clients.add(client_id)
                    
                    # Send processing status
                    await self.manager.send_json(client_id, {
                        "type": "status",
                        "status": "processing",
                        "message": "Processing your audio..."
                    })
                    
                    # Process the audio
                    start_time = time.time()
                    response = await self.process_audio(data, client_id)
                    processing_time = time.time() - start_time
                    logger.info(f"Processing completed in {processing_time:.2f}s for client {client_id}")
                    
                    # Add processing time to response
                    response["processing_time"] = round(processing_time, 2)
                    response["type"] = "response"
                    
                    # Send response back to client
                    await websocket.send_json(response)
                    
                except asyncio.TimeoutError:
                    # This is normal when client is waiting between requests
                    pass
                except WebSocketDisconnect:
                    logger.info(f"Client {client_id} disconnected")
                    break
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in client {client_id} message handling: {error_msg}", exc_info=True)
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "error": error_msg,
                            "text": "I encountered an error processing your request."
                        })
                    except:
                        # Client might be disconnected already
                        break
                finally:
                    # Remove client from processing set
                    if client_id in self.processing_clients:
                        self.processing_clients.remove(client_id)
                
        except Exception as e:
            logger.error(f"Error in WebSocket connection for client {client_id}: {str(e)}", exc_info=True)
        finally:
            self.manager.disconnect(client_id)
            
    async def process_audio(self, audio_data: bytes, client_id: str) -> Dict[str, Any]:
        """Process audio data and generate response"""
        try:
            # Save audio data to temporary file
            temp_file = os.path.join(self.temp_dir, f"temp_{client_id}_{int(time.time())}.webm")
            with open(temp_file, "wb") as f:
                f.write(audio_data)
                
            logger.info(f"Saved audio data to {temp_file} ({len(audio_data)} bytes)")
                
            # Process the audio file
            response = await self.simulator.process_audio(temp_file)
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
                logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
            
            return response
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error processing audio: {str(e)}\n{error_traceback}")
            return {
                "text": "I'm sorry, I had trouble understanding that. Could you please try again?",
                "error": str(e)
            }
            
    async def process_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Process an audio file and generate response"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("Audio file is empty")
                
            logger.info(f"Processing audio file: {file_path} ({file_size} bytes)")
            
            # Process the audio file using the simulator
            start_time = time.time()
            response = await self.simulator.process_audio(file_path)
            processing_time = time.time() - start_time
            
            logger.info(f"Processed audio file in {processing_time:.2f}s")
            response["processing_time"] = round(processing_time, 2)
            
            return response
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error processing audio file: {str(e)}\n{error_traceback}")
            return {
                "text": "I'm sorry, I had trouble processing that audio file. Please try again.",
                "error": str(e)
            } 