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
            check=False,  # Don't raise exception to handle errors ourselves
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

        # Run whisper transcription with verbose output and simplified options
        logger.info("Starting transcription with whisper.cpp")
        
        # The output directory for the transcript
        output_dir = "/tmp"
        txt_file = os.path.join(output_dir, os.path.basename(output_wav).replace(".wav", ".txt"))
        
        # First, try the verbose mode
        whisper_command = [
            WHISPER_BINARY,
            "-m", MODEL_PATH,
            "-f", output_wav,
            "-otxt",
            "--output-dir", output_dir,
            "-v"  # Add verbose flag
        ]
        
        logger.info(f"Running whisper command: {' '.join(whisper_command)}")
        
        whisper_result = subprocess.run(
            whisper_command, 
            check=False,  # Don't raise exception to handle errors ourselves
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        logger.info(f"Whisper process exit code: {whisper_result.returncode}")
        logger.info(f"Whisper stdout: {whisper_result.stdout}")
        
        if whisper_result.returncode != 0:
            logger.error(f"Whisper transcription failed with error: {whisper_result.stderr}")
            
            # Try an alternative approach with simpler options if the first attempt failed
            logger.info("Trying alternative whisper command without extra options")
            simple_whisper_command = [
                WHISPER_BINARY,
                "-m", MODEL_PATH,
                "-f", output_wav
            ]
            
            simple_result = subprocess.run(
                simple_whisper_command, 
                check=False,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            logger.info(f"Simple whisper command exit code: {simple_result.returncode}")
            logger.info(f"Simple whisper stdout: {simple_result.stdout}")
            
            if simple_result.returncode != 0:
                logger.error(f"Simple whisper command also failed: {simple_result.stderr}")
                return JSONResponse(
                    content={
                        "transcript": f"Error: Whisper transcription failed.\nFirst error: {whisper_result.stderr}\nSecond error: {simple_result.stderr}"
                    }, 
                    status_code=500
                )
            else:
                # Use stdout as transcript if the simple command worked
                transcript = simple_result.stdout.strip()
                logger.info(f"Got transcript from stdout: {transcript[:100]}...")
                return {"transcript": transcript}
        
        # If we get here, the first command succeeded, check for the transcript file
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
    """Health check endpoint"""
    logger.info("Health check requested")
    
    binary_exists = os.path.exists(WHISPER_BINARY)
    is_executable = os.access(WHISPER_BINARY, os.X_OK) if binary_exists else False
    model_exists = os.path.exists(MODEL_PATH)
    
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
    
    # Check model file size
    model_size = None
    if model_exists:
        model_size = os.path.getsize(MODEL_PATH)
    
    response = {
        "status": "healthy",
        "whisper_binary_exists": binary_exists,
        "whisper_binary_executable": is_executable,
        "whisper_binary_path": WHISPER_BINARY,
        "binary_info": binary_info,
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "model_size_bytes": model_size,
        "version": "2.1.0"
    }
    
    logger.info(f"Health check response: {response}")
    return response

@app.get("/")
async def root():
    """Root endpoint for quick testing"""
    return {"message": "Whisper API is running", "version": "2.1.0"}
