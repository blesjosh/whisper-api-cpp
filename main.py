import uuid
import os
import subprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil

app = FastAPI()

class TranscriptResponse(BaseModel):
    transcript: str

@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        input_ext = os.path.splitext(file.filename)[1]
        input_path = f"audio_input_{uuid.uuid4()}{input_ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert to 16kHz mono WAV using ffmpeg
        wav_path = f"audio_converted_{uuid.uuid4()}.wav"
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            wav_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Transcribe using whisper.cpp
        whisper_cmd = [
            "./whisper.cpp/bin/main",  # ✅ fixed path
            "-m", "./whisper.cpp/models/base.en.bin",
            "-f", wav_path,
            "-otxt"
        ]
        subprocess.run(whisper_cmd, check=True)

        # Read transcript file
        txt_file = wav_path.replace(".wav", ".txt")
        with open(txt_file, "r") as f:
            transcript = f.read()

        # Clean up
        os.remove(input_path)
        os.remove(wav_path)
        os.remove(txt_file)

        return {"transcript": transcript}

    except Exception as e:
        return JSONResponse(status_code=500, content={"transcript": f"Internal server error:\n{str(e)}"})
