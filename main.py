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

# Use the correct binary paths
WHISPER_CPP_DIR = "/app/whisper.cpp"
WHISPER_BINARY = os.path.join(WHISPER_CPP_DIR, "build/bin/main")
MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models/base.en.bin")

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

        # Convert audio to WAV with detailed ffmpeg output
        logger.info(f"Converting audio file to WAV format")
        
        # First, get info about the input file
        probe_command = [
            "ffmpeg", "-i", input_filename
        ]
        try:
            probe_result = subprocess.run(
                probe_command, 
                check=False,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            logger.info(f"Input file info: {probe_result.stderr}")
        except Exception as e:
            logger.error(f"Error probing input file: {str(e)}")
        
        # Convert the file
        convert_command = [
            "ffmpeg", "-i", input_filename, 
            "-ar", "16000", 
            "-ac", "1", 
            "-c:a", "pcm_s16le",  # Explicitly set codec to PCM 16-bit
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
        
        if os.path.exists(output_wav):
            file_size = os.path.getsize(output_wav)
            logger.info(f"WAV file created successfully: {file_size} bytes")
            
            # Verify the WAV file with ffprobe
            probe_wav_command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", output_wav]
            probe_wav_result = subprocess.run(
                probe_wav_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"WAV file format info: {probe_wav_result.stdout}")
            
            if "pcm_s16le" not in probe_wav_result.stdout:
                logger.warning("WAV file may not be in the expected PCM 16-bit format")
        else:
            logger.error("WAV file was not created")
            return JSONResponse(
                content={"transcript": "Error: Failed to convert audio file"}, 
                status_code=500
            )
            
        # Check model file
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return JSONResponse(
                content={"transcript": f"Error: Model file not found at {MODEL_PATH}"}, 
                status_code=500
            )
        
        model_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Model file size: {model_size} bytes")
        
        # Now run whisper.cpp in multiple ways to ensure we get output
        transcript = ""
        
        # Attempt 1: Standard command
        logger.info("Attempt 1: Running standard whisper command")
        whisper_command = [
            WHISPER_BINARY,
            "-m", MODEL_PATH,
            "-f", output_wav,
            "-otxt"
        ]
        
        logger.info(f"Running whisper command: {' '.join(whisper_command)}")
        
        try:
            whisper_result = subprocess.run(
                whisper_command, 
                check=False,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=30  # Add timeout to prevent hanging
            )
            
            logger.info(f"Whisper process exit code: {whisper_result.returncode}")
            logger.info(f"Whisper stdout: {whisper_result.stdout}")
            logger.info(f"Whisper stderr: {whisper_result.stderr}")
            
            # Check for transcript in output file
            txt_file = output_wav.replace(".wav", ".txt")
            if os.path.exists(txt_file):
                with open(txt_file, "r") as f:
                    transcript = f.read().strip()
                logger.info(f"Found transcript in file {txt_file}, length: {len(transcript)} chars")
            
            # If transcript is empty, try to extract from stdout
            if not transcript and whisper_result.stdout:
                transcript = whisper_result.stdout.strip()
                logger.info(f"Extracted transcript from stdout, length: {len(transcript)} chars")
                
        except subprocess.TimeoutExpired:
            logger.error("Whisper process timed out after 30 seconds")
        
        # Attempt 2: If still no transcript, try with -l auto parameter
        if not transcript:
            logger.info("Attempt 2: Running whisper with language auto-detection")
            whisper_command_2 = [
                WHISPER_BINARY,
                "-m", MODEL_PATH,
                "-f", output_wav,
                "-l", "auto",  # Auto-detect language
                "-otxt"
            ]
            
            try:
                whisper_result_2 = subprocess.run(
                    whisper_command_2, 
                    check=False,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    timeout=30
                )
                
                logger.info(f"Whisper process 2 exit code: {whisper_result_2.returncode}")
                logger.info(f"Whisper stdout 2: {whisper_result_2.stdout}")
                
                # Check for transcript in output file again
                if os.path.exists(txt_file):
                    with open(txt_file, "r") as f:
                        transcript = f.read().strip()
                    logger.info(f"Found transcript in file after attempt 2, length: {len(transcript)} chars")
                
                # If still empty, try stdout again
                if not transcript and whisper_result_2.stdout:
                    # Try to extract transcript from stdout - it might be buried in the output
                    stdout_lines = whisper_result_2.stdout.strip().split("\n")
                    # Skip header lines and take the rest
                    content_lines = [line for line in stdout_lines if line and not line.startswith("[")]
                    if content_lines:
                        transcript = "\n".join(content_lines)
                        logger.info(f"Extracted transcript from stdout lines, length: {len(transcript)} chars")
            except subprocess.TimeoutExpired:
                logger.error("Whisper process 2 timed out after 30 seconds")
        
        # Attempt 3: Try with direct stdout capture mode
        if not transcript:
            logger.info("Attempt 3: Running whisper with direct stdout capture")
            whisper_command_3 = [
                WHISPER_BINARY,
                "-m", MODEL_PATH,
                "-f", output_wav,
                "-l", "en",  # Force English
                "--print-special",  # Print special tokens
                "--print-progress"  # Show progress
            ]
            
            try:
                whisper_result_3 = subprocess.run(
                    whisper_command_3, 
                    check=False,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    timeout=30
                )
                
                logger.info(f"Whisper process 3 exit code: {whisper_result_3.returncode}")
                
                # Extract transcript from detailed output
                output = whisper_result_3.stdout
                logger.info(f"Full stdout from attempt 3: {output}")
                
                if output:
                    # Process the output to extract the transcript
                    lines = output.split('\n')
                    content_lines = []
                    for line in lines:
                        # Skip timestamp lines and progress indicators
                        if line and not line.startswith('[') and not ': ' in line:
                            content_lines.append(line)
                    
                    if content_lines:
                        transcript = ' '.join(content_lines).strip()
                        logger.info(f"Extracted processed transcript, length: {len(transcript)} chars")
            except subprocess.TimeoutExpired:
                logger.error("Whisper process 3 timed out after 30 seconds")
        
        # Cleanup temp files
        if background_tasks:
            background_tasks.add_task(cleanup_temp_files, temp_dir)
        else:
            # Immediate cleanup
            cleanup_temp_files(temp_dir)
        
        if not transcript:
            logger.warning("All attempts produced empty transcripts")
            return {"transcript": "Could not generate transcript. The audio might be silent or in an unsupported format."}
            
        return {"transcript": transcript}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in transcription: {str(e)}")
        logger.error(error_details)
        
        # Cleanup on error
        try:
            if os.path.exists(temp_dir):
                cleanup_temp_files(temp_dir)
        except:
            pass
            
        return JSONResponse(
            content={"transcript": f"Internal server error:\n{str(e)}\n\nDetails:\n{error_details}"}, 
            status_code=500
        )

def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

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
    
    # Run a simple test transcription on a silent audio file
    test_result = "Not run"
    try:
        # Create a small silent WAV file for testing
        test_dir = tempfile.mkdtemp()
        test_wav = os.path.join(test_dir, "test.wav")
        
        create_silent_wav_cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", 
            "-t", "1", "-c:a", "pcm_s16le", test_wav
        ]
        subprocess.run(create_silent_wav_cmd, check=True, capture_output=True)
        
        # Run whisper on the test file
        test_cmd = [WHISPER_BINARY, "-m", MODEL_PATH, "-f", test_wav]
        test_proc = subprocess.run(test_cmd, check=False, capture_output=True, text=True, timeout=10)
        
        test_result = {
            "exit_code": test_proc.returncode,
            "stdout_sample": test_proc.stdout[:200] if test_proc.stdout else "",
            "stderr_sample": test_proc.stderr[:200] if test_proc.stderr else ""
        }
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        test_result = f"Test failed: {str(e)}"
    
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
        "test_transcription": test_result,
        "version": "3.1.0"
    }
    
    logger.info(f"Health check response: {json.dumps(response, indent=2)}")
    return response

@app.get("/")
async def root():
    """Root endpoint for quick testing"""
    return {"message": "Whisper API is running", "version": "3.1.0"}
