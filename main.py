from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import subprocess
import os
import uuid

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the original file with its extension
    original_ext = os.path.splitext(file.filename)[-1]
    temp_input_path = f"audio_input_{uuid.uuid4()}{original_ext}"
    temp_wav_path = f"audio_converted_{uuid.uuid4()}.wav"

    print(f"Received file: {file.filename}")
    print(f"Saving input as: {temp_input_path}")
    print(f"Will convert to WAV: {temp_wav_path}")

    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Convert any audio format to WAV (16kHz mono) using ffmpeg
        print("Running ffmpeg conversion...")
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input_path,
            "-ar", "16000", "-ac", "1", temp_wav_path
        ], check=True)
        print("FFmpeg conversion complete.")

        # Run transcription using whisper.cpp
        print("Running whisper.cpp transcription...")
        result = subprocess.run(
            ["./main", "-m", "models/base.en.bin", "-f", temp_wav_path, "-otxt"],
            capture_output=True,
            text=True,
            timeout=120
        )

        print("Whisper stdout:", result.stdout)
        print("Whisper stderr:", result.stderr)

        transcript_path = temp_wav_path.replace(".wav", ".txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                transcript = f.read()
        else:
            print("Transcript file not found.")
            transcript = "Transcription failed or file not found."

    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e.stderr)
        transcript = f"FFmpeg or transcription error: {e.stderr}"

    except Exception as e:
        print("Unhandled exception:", str(e))
        transcript = str(e)

    finally:
        # Clean up all temp files
        for path in [temp_input_path, temp_wav_path, transcript_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")

    return JSONResponse({"transcript": transcript})
