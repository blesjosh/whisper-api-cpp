import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

# Use absolute paths
WHISPER_BINARY = "/app/whisper.cpp/build/bin/main"
MODEL_PATH = "/app/whisper.cpp/models/base.en.bin"

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Generate unique filenames
        input_filename = f"/tmp/audio_input_{uuid.uuid4()}.opus"
        output_wav = f"/tmp/audio_converted_{uuid.uuid4()}.wav"
        
        # Save uploaded file
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert audio to WAV
        convert_command = [
            "ffmpeg", "-i", input_filename, "-ar", "16000", "-ac", "1", "-f", "wav", output_wav
        ]
        subprocess.run(convert_command, check=True)

        # Debug information
        app.logger.info(f"Looking for whisper binary at: {WHISPER_BINARY}")
        if os.path.exists(WHISPER_BINARY):
            app.logger.info(f"Binary exists")
        else:
            app.logger.info(f"Binary not found")
            # Try to list what's in the directory
            dir_path = os.path.dirname(WHISPER_BINARY)
            if os.path.exists(dir_path):
                app.logger.info(f"Contents of {dir_path}: {os.listdir(dir_path)}")
        
        # Run whisper transcription with absolute paths
        whisper_command = [
            WHISPER_BINARY,
            "-m", MODEL_PATH,
            "-f", output_wav,
            "-otxt"
        ]
        subprocess.run(whisper_command, check=True)

        # Read the transcript
        txt_file = output_wav.replace(".wav", ".txt")
        with open(txt_file, "r") as f:
            transcript = f.read()

        # Cleanup
        os.remove(input_filename)
        os.remove(output_wav)
        os.remove(txt_file)

        return {"transcript": transcript}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            content={"transcript": f"Internal server error:\n{str(e)}\n\nDetails:\n{error_details}"}, 
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Simple health check that also verifies the whisper binary exists"""
    if os.path.exists(WHISPER_BINARY):
        binary_exists = True
        # Check if it's executable
        is_executable = os.access(WHISPER_BINARY, os.X_OK)
    else:
        binary_exists = False
        is_executable = False
    
    return {
        "status": "healthy",
        "whisper_binary_exists": binary_exists,
        "whisper_binary_executable": is_executable,
        "whisper_binary_path": WHISPER_BINARY,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }
