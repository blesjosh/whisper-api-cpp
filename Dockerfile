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

WORKDIR /app

# First, copy your FastAPI code
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone and build whisper.cpp (AFTER copying so it's not overwritten)
RUN git clone https://github.com/ggerganov/whisper.cpp.git \
    && cd whisper.cpp && make

# Run build script (if needed)
RUN chmod +x build.sh && ./build.sh

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
