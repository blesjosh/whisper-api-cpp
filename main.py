import os
import uuid
import subprocess
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

app = FastAPI()

# Use absolute paths
WHISPER_BINARY = "/app/whisper.cpp/build/bin/main"
MODEL_PATH = "/app/whisper.cpp/models/base.en.bin"

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    logger.info("Transcription request received")
    try:
        # Generate unique filenames
        input_filename = f"/tmp/audio_input_{uuid.uuid4()}.opus"
        output_wav = f"/tmp/audio_converted_{uuid.uuid4()}.wav"
        
        # Save uploaded file
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Saved uploaded file to {input_filename}")

        # Convert audio to WAV
        logger.info(f"Converting audio file to WAV format")
        convert_command = [
            "ffmpeg", "-i", input_filename, "-ar", "16000", "-ac", "1", "-f", "wav", output_wav
        ]
        convert_result = subprocess.run(
            convert_command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if os.path.exists(output_wav):
            file_size = os.path.getsize(output_wav)
            logger.info(f"WAV file created successfully: {file_size} bytes")
        else:
            logger.error("WAV file was not created")
            return JSONResponse(
                content={"transcript": "Error: Failed to convert audio file"}, 
                status_code=500
            )

        # Check whisper binary
        logger.info(f"Checking whisper binary at {WHISPER_BINARY}")
        if not os.path.exists(WHISPER_BINARY):
            logger.error(f"Whisper binary not found at {WHISPER_BINARY}")
            dir_path = os.path.dirname(WHISPER_BINARY)
            if os.path.exists(dir_path):
                logger.info(f"Contents of {dir_path}: {os.listdir(dir_path)}")
            return JSONResponse(
                content={"transcript": f"Error: Whisper binary not found at {WHISPER_BINARY}"}, 
                status_code=500
            )
        
        # Run whisper transcription
        logger.info("Starting transcription with whisper.cpp")
        whisper_command = [
            WHISPER_BINARY,
            "-m", MODEL_PATH,
            "-f", output_wav,
            "-otxt"
        ]
        whisper_result = subprocess.run(
            whisper_command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        logger.info("Whisper transcription process completed")
        
        # Read the transcript
        txt_file = output_wav.replace(".wav", ".txt")
        logger.info(f"Looking for transcript file at {txt_file}")
        
        if not os.path.exists(txt_file):
            logger.error(f"Transcript file not found at {txt_file}")
            return JSONResponse(
                content={"transcript": "Error: Transcript file not generated"}, 
                status_code=500
            )
            
        with open(txt_file, "r") as f:
            transcript = f.read()
            
        logger.info(f"Successfully read transcript: {len(transcript)} characters")

        # Cleanup
        os.remove(input_filename)
        os.remove(output_wav)
        os.remove(txt_file)
        logger.info("Temporary files cleaned up")

        return {"transcript": transcript}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in transcription: {str(e)}")
        logger.error(error_details)
        return JSONResponse(
            content={"transcript": f"Internal server error:\n{str(e)}\n\nDetails:\n{error_details}"}, 
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    
    binary_exists = os.path.exists(WHISPER_BINARY)
    is_executable = os.access(WHISPER_BINARY, os.X_OK) if binary_exists else False
    model_exists = os.path.exists(MODEL_PATH)
    
    response = {
        "status": "healthy",
        "whisper_binary_exists": binary_exists,
        "whisper_binary_executable": is_executable,
        "whisper_binary_path": WHISPER_BINARY,
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "version": "2.0.0"  # Add a version to track updates
    }
    
    logger.info(f"Health check response: {response}")
    return response

@app.get("/")
async def root():
    """Root endpoint for quick testing"""
    return {"message": "Whisper API is running", "version": "2.0.0"}
