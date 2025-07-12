from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import subprocess
import os
import uuid

app = FastAPI()

# ✅ Root route to show app is live
@app.get("/")
def root():
    return {"message": "Whisper API is live!"}

# 🎙️ Transcription route
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_filename = f"audio_{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = subprocess.run(
            ["./main", "-m", "models/base.en.bin", "-f", temp_filename, "-otxt"],
            capture_output=True,
            text=True,
            timeout=120
        )

        transcript_path = temp_filename.replace(".wav", ".txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                transcript = f.read()
        else:
            transcript = "Transcription failed or file not found."

    finally:
        # Clean up
        os.remove(temp_filename)
        if os.path.exists(transcript_path):
            os.remove(transcript_path)

    return JSONResponse({"transcript": transcript})
