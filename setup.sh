#!/data/data/com.termux/files/usr/bin/bash

pkg update -y
pkg install -y git cmake make clang python ffmpeg termux-api portaudio
pkg install -y python-numpy

pip install -r requirements.txt

if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp
fi

cd whisper.cpp || exit 1

cmake -B build
cmake --build build -j$(nproc)

cd ..

chmod +x download_model.sh
./download_model.sh
