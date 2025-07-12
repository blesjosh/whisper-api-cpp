import os
import uuid
import subprocess
import logging
import json
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

# Use the correct binary (whisper or main depending on version)
WHISPER_CPP_DIR = "/app/whisper.cpp"
WHISPER_BINARY = os.path.join(WHISPER_CPP_DIR, "build/bin/main")  # Default to old path
MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models/base.en.bin")

# Check if the newer binary exists and use it instead
WHISPER_CLI_PATH = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
if os.path.exists(WHISPER_CLI_PATH) and os.access(WHISPER_CLI_PATH, os.X_OK):
    WHISPER_BINARY = WHISPER_CLI_PATH
    logger.info(f"Using newer whisper-cli binary: {WHISPER_BINARY}")
else:
    logger.info(f"Using legacy main binary: {WHISPER_BINARY}")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    logger.info("Transcription request received")
    try:
        # First, verify the model file is valid
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return JSONResponse(
                content={"transcript": f"Error: Model file not found at {MODEL_PATH}"}, 
                status_code=500
            )
        
        model_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Model file size: {model_size} bytes")
        
        if model_size < 100000000:  # Less than 100MB
            logger.error(f"Model file is too small ({model_size} bytes). It appears to be corrupted or incomplete.")
            return JSONResponse(
                content={"transcript": f"Error: Model file is corrupted or incomplete ({model_size} bytes)"}, 
                status_code=500
            )
        
        # Generate unique filenames with full paths
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
            check=False,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if convert_result.returncode != 0:
            logger.error(f"ffmpeg conversion failed with error: {convert_result.stderr}")
            return JSONResponse(
                content={"transcript": f"Error: Failed to convert audio file. ffmpeg error: {convert_result.stderr}"}, 
                status_code=500
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

        # Run whisper transcription
        logger.info("Starting transcription with whisper.cpp")
        output_dir = "/tmp"
        txt_file = os.path.join(output_dir, os.path.basename(output_wav).replace(".wav", ".txt"))
        
        # Use command format based on binary name
        if "whisper-cli" in WHISPER_BINARY:
            # New whisper-cli syntax
            whisper_command = [
                WHISPER_BINARY,
                "--model", MODEL_PATH,
                "--file", output_wav,
                "--output-txt",
                "--output-file", txt_file,
                "--verbose"
            ]
        else:
            # Old main binary syntax
            whisper_command = [
                WHISPER_BINARY,
                "-m", MODEL_PATH,
                "-f", output_wav,
                "-otxt",
                "--output-dir", output_dir
            ]
        
        logger.info(f"Running whisper command: {' '.join(whisper_command)}")
        
        whisper_result = subprocess.run(
            whisper_command, 
            check=False,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        logger.info(f"Whisper process exit code: {whisper_result.returncode}")
        logger.info(f"Whisper stdout: {whisper_result.stdout}")
        logger.info(f"Whisper stderr: {whisper_result.stderr}")
        
        if whisper_result.returncode != 0:
            logger.error(f"Whisper transcription failed with exit code {whisper_result.returncode}")
            return JSONResponse(
                content={
                    "transcript": f"Error: Whisper transcription failed with exit code {whisper_result.returncode}.\n" +
                                 f"Stdout: {whisper_result.stdout}\n" +
                                 f"Stderr: {whisper_result.stderr}"
                }, 
                status_code=500
            )
        
        # Check for the transcript file
        logger.info(f"Looking for transcript file at {txt_file}")
        
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                transcript = f.read()
            logger.info(f"Successfully read transcript from file: {len(transcript)} characters")
        else:
            # If no file but process succeeded, try using stdout
            logger.info("Transcript file not found, using stdout instead")
            transcript = whisper_result.stdout.strip()
            logger.info(f"Got transcript from stdout: {transcript[:100]}...")
            
        # Cleanup
        try:
            os.remove(input_filename)
            os.remove(output_wav)
            if os.path.exists(txt_file):
                os.remove(txt_file)
            logger.info("Temporary files cleaned up")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")

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
    """Health check endpoint with extensive diagnostics"""
    logger.info("Health check requested")
    
    # Check binary
    binary_exists = os.path.exists(WHISPER_BINARY)
    is_executable = os.access(WHISPER_BINARY, os.X_OK) if binary_exists else False
    
    # Check model
    model_exists = os.path.exists(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
    model_status = "Valid" if model_size > 100000000 else "Invalid/Corrupted"
    
    # Check binary info
    binary_info = None
    if binary_exists:
        try:
            version_result = subprocess.run(
                [WHISPER_BINARY, "-h"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            binary_info = version_result.stdout or version_result.stderr
        except Exception as e:
            binary_info = f"Error checking binary: {str(e)}"
    
    # List files in model directory
    model_dir = os.path.dirname(MODEL_PATH)
    model_dir_files = []
    if os.path.exists(model_dir):
        try:
            model_dir_files = [
                {"name": f, "size": os.path.getsize(os.path.join(model_dir, f))}
                for f in os.listdir(model_dir)
            ]
        except Exception as e:
            model_dir_files = [{"error": str(e)}]
    
    response = {
        "status": "healthy" if (binary_exists and is_executable and model_exists and model_size > 100000000) else "unhealthy",
        "binary": {
            "path": WHISPER_BINARY,
            "exists": binary_exists,
            "executable": is_executable,
            "info": binary_info
        },
        "model": {
            "path": MODEL_PATH,
            "exists": model_exists,
            "size_bytes": model_size,
            "status": model_status,
            "directory_files": model_dir_files
        },
        "version": "3.0.0"
    }
    
    logger.info(f"Health check response: {json.dumps(response, indent=2)}")
    return response

@app.get("/")
async def root():
    """Root endpoint for quick testing"""
    return {"message": "Whisper API is running", "version": "3.0.0"}
