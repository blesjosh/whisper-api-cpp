import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

# Get absolute base directory (where main.py is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_BIN = os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "main")
MODEL_PATH = os.path.join(BASE_DIR, "whisper.cpp", "models", "base.en.bin")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        input_filename = os.path.join(BASE_DIR, f"audio_input_{uuid.uuid4()}.opus")
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)

        output_wav = os.path.join(BASE_DIR, f"audio_converted_{uuid.uuid4()}.wav")
        convert_command = [
            "ffmpeg", "-i", input_filename, "-ar", "16000", "-ac", "1", "-f", "wav", output_wav
        ]
        subprocess.run(convert_command, check=True)

        whisper_command = [
            WHISPER_BIN,
            "-m", MODEL_PATH,
            "-f", output_wav,
            "-otxt"
        ]
        subprocess.run(whisper_command, check=True)

        txt_file = output_wav.replace(".wav", ".txt")
        with open(txt_file, "r") as f:
            transcript = f.read()

        os.remove(input_filename)
        os.remove(output_wav)
        os.remove(txt_file)

        return {"transcript": transcript}

    except Exception as e:
        return JSONResponse(content={"transcript": f"Internal server error:\n{str(e)}"}, status_code=500)import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

# Get absolute base directory (where main.py is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_BIN = os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "main")
MODEL_PATH = os.path.join(BASE_DIR, "whisper.cpp", "models", "base.en.bin")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        input_filename = os.path.join(BASE_DIR, f"audio_input_{uuid.uuid4()}.opus")
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)

        output_wav = os.path.join(BASE_DIR, f"audio_converted_{uuid.uuid4()}.wav")
        convert_command = [
            "ffmpeg", "-i", input_filename, "-ar", "16000", "-ac", "1", "-f", "wav", output_wav
        ]
        subprocess.run(convert_command, check=True)

        whisper_command = [
            WHISPER_BIN,
            "-m", MODEL_PATH,
            "-f", output_wav,
            "-otxt"
        ]
        subprocess.run(whisper_command, check=True)

        txt_file = output_wav.replace(".wav", ".txt")
        with open(txt_file, "r") as f:
            transcript = f.read()

        os.remove(input_filename)
        os.remove(output_wav)
        os.remove(txt_file)

        return {"transcript": transcript}

    except Exception as e:
        return JSONResponse(content={"transcript": f"Internal server error:\n{str(e)}"}, status_code=500)
