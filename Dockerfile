FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    git

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone and build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git \
    && cd whisper.cpp && make

# Copy API code (main.py, build.sh, etc.)
COPY . .

# Make sure the build script is executable and run it
RUN chmod +x build.sh && ./build.sh

# Expose port (optional for local testing)
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
