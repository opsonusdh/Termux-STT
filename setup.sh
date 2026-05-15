#!/data/data/com.termux/files/usr/bin/bash

pkg update -y
pkg install -y git cmake make clang python ffmpeg termux-api

pip install -r requirements.txt

if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp
fi

cd whisper.cpp

cmake -B build
cmake --build build -j$(nproc)

bash ./models/download-ggml-model.sh tiny.en

cd ..
