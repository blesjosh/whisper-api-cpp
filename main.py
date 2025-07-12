import os
import uuid
import subprocess
import logging
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

app = FastAPI()

# Look for the correct whisper binary
WHISPER_CPP_DIR = "/app/whisper.cpp"
WHISPER_BINARY_MAIN = os.path.join(WHISPER_CPP_DIR, "build/bin/main")
WHISPER_BINARY_CLI = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")

# Check which binary exists and is executable
if os.path.exists(WHISPER_BINARY_CLI) and os.access(WHISPER_BINARY_CLI, os.X_OK):
    WHISPER_BINARY = WHISPER_BINARY_CLI
    logger.info(f"Using whisper-cli binary: {WHISPER_BINARY}")
else:
    WHISPER_BINARY = WHISPER_BINARY_MAIN
    logger.info(f"Using main binary: {WHISPER_BINARY}")

# Try both models - base first, then tiny as fallback
BASE_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models/base.en.bin")
TINY_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models/tiny.en.bin")

if os.path.exists(BASE_MODEL_PATH) and os.path.getsize(BASE_MODEL_PATH) > 100000000:
    MODEL_PATH = BASE_MODEL_PATH
    logger.info(f"Using base model: {MODEL_PATH}")
elif os.path.exists(TINY_MODEL_PATH) and os.path.getsize(TINY_MODEL_PATH) > 10000000:
    MODEL_PATH = TINY_MODEL_PATH
    logger.info(f"Using tiny model: {MODEL_PATH}")
else:
    # Default to base model and let the health check report the issue
    MODEL_PATH = BASE_MODEL_PATH
    logger.warning(f"No valid model found, defaulting to: {MODEL_PATH}")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    logger.info(f"Transcription request received for file: {file.filename}")
    
    # Create a unique temporary directory for this request
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Generate unique filenames with full paths
        input_filename = os.path.join(temp_dir, f"input_{uuid.uuid4()}.{file.filename.split('.')[-1]}")
        output_wav = os.path.join(temp_dir, f"converted_{uuid.uuid4()}.wav")
        
        # Save uploaded file
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Saved uploaded file to {input_filename}, size: {os.path.getsize(input_filename)} bytes")

        # Convert audio to WAV
        logger.info(f"Converting audio file to WAV format")
        convert_command = [
            "ffmpeg", "-i", input_filename, 
            "-ar", "16000", 
            "-ac", "1", 
            "-c:a", "pcm_s16le",
            "-f", "wav", 
            output_wav
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
        
        if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
            logger.error("WAV file was not created or is empty")
            return JSONResponse(
                content={"transcript": "Error: Failed to convert audio file"}, 
                status_code=500
            )
            
        # Try multiple approaches to get a transcript
        transcript = None
        approaches_tried = 0
        
        # First, try with specific command-line options
        logger.info("Running whisper transcription with standard options")
        approaches_tried += 1
        
        if WHISPER_BINARY == WHISPER_BINARY_CLI:
            # New whisper-cli command format
            whisper_cmd = [
                WHISPER_BINARY,
                "--model", MODEL_PATH,
                "--file", output_wav,
                "--output-txt"
            ]
        else:
            # Old main binary format
            whisper_cmd = [
                WHISPER_BINARY,
                "-m", MODEL_PATH,
                "-f", output_wav,
                "-otxt"
            ]
        
        logger.info(f"Running command: {' '.join(whisper_cmd)}")
        
        result = subprocess.run(
            whisper_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Command exit code: {result.returncode}")
        logger.info(f"Command stdout: {result.stdout}")
        logger.info(f"Command stderr: {result.stderr}")
        
        # Look for the transcript
        txt_file = output_wav.replace(".wav", ".txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                transcript = f.read().strip()
            logger.info(f"Found transcript in file, length: {len(transcript)}")
        
        # If no transcript in file, try to extract from stdout
        if not transcript:
            # Skip the warning message about deprecated binary
            stdout_lines = result.stdout.split("\n")
            filtered_lines = []
            
            for line in stdout_lines:
                # Skip the deprecation warning
                if any(warning in line for warning in [
                    "WARNING:", "deprecated", "whisper-cli", "https://github.com"
                ]):
                    continue
                if line.strip() == "":
                    continue
                
                # Keep substantive content
                filtered_lines.append(line)
            
            if filtered_lines:
                transcript = "\n".join(filtered_lines)
                logger.info(f"Extracted transcript from stdout, length: {len(transcript)}")
        
        # If still no transcript, try with streaming output
        if not transcript:
            logger.info("Trying alternative approach with streaming output")
            approaches_tried += 1
            
            if WHISPER_BINARY == WHISPER_BINARY_CLI:
                stream_cmd = [
                    WHISPER_BINARY,
                    "--model", MODEL_PATH,
                    "--file", output_wav,
                    "--print-special",
                    "--print-progress"
                ]
            else:
                stream_cmd = [
                    WHISPER_BINARY,
                    "-m", MODEL_PATH,
                    "-f", output_wav,
                    "--print-special",
                    "--print-progress"
                ]
            
            stream_result = subprocess.run(
                stream_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Stream command exit code: {stream_result.returncode}")
            
            # Filter out the warnings and extract actual content
            stdout_content = stream_result.stdout
            if stdout_content:
                # Skip warning lines and extract actual content
                lines = stdout_content.split('\n')
                content_lines = []
                for line in lines:
                    # Skip deprecation warning and empty lines
                    if any(warning in line for warning in [
                        "WARNING:", "deprecated", "whisper-cli", "https://github.com"
                    ]):
                        continue
                    
                    # Skip progress indicators
                    if line.startswith('['):
                        continue
                        
                    # Keep non-empty lines that aren't part of the warning
                    if line.strip():
                        content_lines.append(line)
                
                if content_lines:
                    transcript = "\n".join(content_lines)
                    logger.info(f"Extracted transcript from streaming output, length: {len(transcript)}")
        
        # Try one more approach with language specification
        if not transcript:
            logger.info("Trying approach with explicit language setting")
            approaches_tried += 1
            
            if WHISPER_BINARY == WHISPER_BINARY_CLI:
                lang_cmd = [
                    WHISPER_BINARY,
                    "--model", MODEL_PATH,
                    "--file", output_wav,
                    "--language", "en"
                ]
            else:
                lang_cmd = [
                    WHISPER_BINARY,
                    "-m", MODEL_PATH,
                    "-f", output_wav,
                    "-l", "en"
                ]
            
            lang_result = subprocess.run(
                lang_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Language-specific command exit code: {lang_result.returncode}")
            
            # Process the output
            if lang_result.stdout:
                lines = lang_result.stdout.split('\n')
                content_lines = []
                for line in lines:
                    # Skip warnings and progress indicators
                    if any(warning in line for warning in [
                        "WARNING:", "deprecated", "whisper-cli", "https://github.com"
                    ]) or line.startswith('[') or not line.strip():
                        continue
                    content_lines.append(line)
                
                if content_lines:
                    transcript = "\n".join(content_lines)
                    logger.info(f"Extracted transcript from language-specific command, length: {len(transcript)}")
            
        # Try tiny model as fallback if base model failed
        if not transcript and MODEL_PATH == BASE_MODEL_PATH and os.path.exists(TINY_MODEL_PATH):
            logger.info("Trying tiny model as fallback")
            approaches_tried += 1
            
            if WHISPER_BINARY == WHISPER_BINARY_CLI:
                tiny_cmd = [
                    WHISPER_BINARY,
                    "--model", TINY_MODEL_PATH,
                    "--file", output_wav
                ]
            else:
                tiny_cmd = [
                    WHISPER_BINARY,
                    "-m", TINY_MODEL_PATH,
                    "-f", output_wav
                ]
            
            tiny_result = subprocess.run(
                tiny_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Tiny model command exit code: {tiny_result.returncode}")
            
            # Process the output
            if tiny_result.stdout:
                lines = tiny_result.stdout.split('\n')
                content_lines = []
                for line in lines:
                    # Skip warnings and progress indicators
                    if any(warning in line for warning in [
                        "WARNING:", "deprecated", "whisper-cli", "https://github.com"
                    ]) or line.startswith('[') or not line.strip():
                        continue
                    content_lines.append(line)
                
                if content_lines:
                    transcript = "\n".join(content_lines)
                    logger.info(f"Extracted transcript from tiny model, length: {len(transcript)}")
        
        # Clean up
        if background_tasks:
            background_tasks.add_task(cleanup_temp_files, temp_dir)
        else:
            cleanup_temp_files(temp_dir)
        
        # If we still have no transcript, return a helpful message
        if not transcript:
            logger.warning(f"No transcript could be extracted after {approaches_tried} attempts")
            return {"transcript": "Could not generate transcript. The audio might be silent, in an unsupported format, or the models might be having issues."}
        
        return {"transcript": transcript}
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in transcription: {str(e)}")
        logger.error(error_details)
        
        # Clean up on error
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
            
        return JSONResponse(
            content={"transcript": f"Internal server error:\n{str(e)}\n\nDetails:\n{error_details}"}, 
            status_code=500
        )

def cleanup_temp_files(temp_dir):
    """Clean up temporary directory and its contents"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@app.get("/test-whisper")
async def test_whisper():
    """Test the whisper binary directly with a generated test file"""
    temp_dir = tempfile.mkdtemp()
    test_wav = os.path.join(temp_dir, "test.wav")
    
    try:
        # Create a simple test WAV file with a tone
        create_test_cmd = [
            "ffmpeg", "-f", "lavfi", 
            "-i", "sine=frequency=1000:duration=3", 
            "-ar", "16000", "-ac", "1", 
            test_wav
        ]
        
        subprocess.run(create_test_cmd, check=True, capture_output=True)
        
        # Run whisper on the test file and capture all output
        if WHISPER_BINARY == WHISPER_BINARY_CLI:
            test_whisper_cmd = [
                WHISPER_BINARY,
                "--model", MODEL_PATH,
                "--file", test_wav,
                "--output-txt"
            ]
        else:
            test_whisper_cmd = [
                WHISPER_BINARY,
                "-m", MODEL_PATH,
                "-f", test_wav,
                "-otxt"
            ]
        
        test_result = subprocess.run(
            test_whisper_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check for output file
        txt_file = test_wav.replace(".wav", ".txt")
        txt_content = None
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                txt_content = f.read()
        
        result = {
            "command": " ".join(test_whisper_cmd),
            "exit_code": test_result.returncode,
            "stdout": test_result.stdout,
            "stderr": test_result.stderr,
            "txt_file_exists": os.path.exists(txt_file),
            "txt_content": txt_content,
            "binary_path": WHISPER_BINARY,
            "model_path": MODEL_PATH,
            "model_size": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        }
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return result
    
    except Exception as e:
        import traceback
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/health")
async def health_check():
    """Health check endpoint with extensive diagnostics"""
    logger.info("Health check requested")
    
    # Check binary
    binary_exists = os.path.exists(WHISPER_BINARY)
    is_executable = os.access(WHISPER_BINARY, os.X_OK) if binary_exists else False
    
    # Check available binaries
    main_exists = os.path.exists(WHISPER_BINARY_MAIN)
    main_executable = os.access(WHISPER_BINARY_MAIN, os.X_OK) if main_exists else False
    cli_exists = os.path.exists(WHISPER_BINARY_CLI)
    cli_executable = os.access(WHISPER_BINARY_CLI, os.X_OK) if cli_exists else False
    
    # Check models
    base_model_exists = os.path.exists(BASE_MODEL_PATH)
    base_model_size = os.path.getsize(BASE_MODEL_PATH) if base_model_exists else 0
    base_model_status = "Valid" if base_model_size > 100000000 else "Invalid/Corrupted"
    
    tiny_model_exists = os.path.exists(TINY_MODEL_PATH)
    tiny_model_size = os.path.getsize(TINY_MODEL_PATH) if tiny_model_exists else 0
    tiny_model_status = "Valid" if tiny_model_size > 10000000 else "Invalid/Corrupted"
    
    # Check active model
    model_exists = os.path.exists(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
    model_status = "Valid" if model_size > 10000000 else "Invalid/Corrupted"
    
    # Basic health check response
    response = {
        "status": "healthy" if (binary_exists and is_executable and model_exists and model_size > 10000000) else "unhealthy",
        "binary": {
            "path": WHISPER_BINARY,
            "exists": binary_exists,
            "executable": is_executable,
            "type": "whisper-cli" if WHISPER_BINARY == WHISPER_BINARY_CLI else "main"
        },
        "available_binaries": {
            "main": {
                "path": WHISPER_BINARY_MAIN,
                "exists": main_exists,
                "executable": main_executable
            },
            "whisper-cli": {
                "path": WHISPER_BINARY_CLI,
                "exists": cli_exists,
                "executable": cli_executable
            }
        },
        "models": {
            "active_model": {
                "path": MODEL_PATH,
                "exists": model_exists,
                "size_bytes": model_size,
                "status": model_status
            },
            "base": {
                "path": BASE_MODEL_PATH,
                "exists": base_model_exists,
                "size_bytes": base_model_size,
                "status": base_model_status
            },
            "tiny": {
                "path": TINY_MODEL_PATH,
                "exists": tiny_model_exists,
                "size_bytes": tiny_model_size,
                "status": tiny_model_status
            }
        },
        "version": "4.1.0"
    }
    
    return response

@app.get("/")
async def root():
    """Root endpoint for quick testing"""
    return {"message": "Whisper API is running", "version": "4.1.0"}