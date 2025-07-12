from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import subprocess
import os
import uuid

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    original_ext = os.path.splitext(file.filename)[-1]
    temp_input_path = f"audio_input_{uuid.uuid4()}{original_ext}"
    temp_wav_path = f"audio_converted_{uuid.uuid4()}.wav"
    transcript_path = f"{temp_wav_path}.txt"  # explicitly define the output path

    # Save uploaded file to disk
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Convert any audio to 16kHz mono WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input_path,
            "-ar", "16000", "-ac", "1", temp_wav_path
        ], check=True)

        # Run whisper.cpp transcription
        result = subprocess.run(
            [
                "./whisper.cpp/build/bin/main",
                "-m", "models/base.en.bin",
                "-f", temp_wav_path,
                "-otxt",
                "-of", transcript_path  # specify exact output location
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Optional: Debug output
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

        # Read transcript
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                transcript = f.read()
        else:
            transcript = "Transcription failed or file not found."

    except subprocess.CalledProcessError as e:
        transcript = f"FFmpeg or transcription error:\n{e.stderr}"
    except Exception as e:
        transcript = f"Internal server error:\n{str(e)}"
    finally:
        for path in [temp_input_path, temp_wav_path, transcript_path]:
            if os.path.exists(path):
                os.remove(path)

    return JSONResponse({"transcript": transcript})
