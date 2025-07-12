import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        input_filename = f"audio_input_{uuid.uuid4()}.opus"
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert to wav
        output_wav = f"audio_converted_{uuid.uuid4()}.wav"
        convert_command = [
            "ffmpeg",
            "-i", input_filename,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            output_wav
        ]
        subprocess.run(convert_command, check=True)

        # Transcribe using whisper.cpp binary (corrected path)
        whisper_command = [
            "./whisper.cpp/build/bin/main",
            "-m", "./whisper.cpp/models/base.en.bin",
            "-f", output_wav,
            "-otxt"
        ]
        subprocess.run(whisper_command, check=True)

        # Read transcript
        txt_file = output_wav.replace(".wav", ".txt")
        with open(txt_file, "r") as f:
            transcript = f.read()

        # Cleanup
        os.remove(input_filename)
        os.remove(output_wav)
        os.remove(txt_file)

        return {"transcript": transcript}

    except Exception as e:
        return JSONResponse(
            content={"transcript": f"Internal server error:\n{str(e)}"},
            status_code=500
        )
