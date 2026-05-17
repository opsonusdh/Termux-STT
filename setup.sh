#!/data/data/com.termux/files/usr/bin/bash
# FIX: exit immediately on any error so a failed build does not silently
# continue to download_model.sh and leave the user with a model but no binary.
set -euo pipefail

pkg update -y
pkg install -y git cmake make clang python ffmpeg termux-api portaudio
pkg install -y python-numpy

pip install -r requirements.txt --break-system-packages

if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp
fi

cd whisper.cpp || exit 1

# FIX: build in Release mode for significantly faster inference on Android.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

cd ..

chmod +x download_model.sh
./download_model.sh
