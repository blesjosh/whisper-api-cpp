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

# Copy your entire app including models and whisper.cpp if already cloned locally
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Build whisper.cpp
RUN cd whisper.cpp && make

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
