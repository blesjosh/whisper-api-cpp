#!/bin/bash
mkdir -p models
curl -L -o models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/models/ggml-base.en.bin
