#!/data/data/com.termux/files/usr/bin/bash

set -euo pipefail


pkg update -y
pkg upgrade -y

pkg install -y \
    git cmake make clang \
    python python-numpy \
    ffmpeg termux-api portaudio which

pip install -r requirements.txt --break-system-packages

echo -e "\nTesting microphone permission.\nAccept the popup if appeared."
TMP_MIC_FILE="$HOME/.termux_stt_mic_test.wav"

termux-microphone-record -f "$TMP_MIC_FILE" -l 1

sleep 2

rm -f "$TMP_MIC_FILE"

if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp
fi

cd whisper.cpp || exit 1
rm -rf build


cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native"

cmake --build build -j "$(nproc)"

echo ""
cd ..

chmod +x download_model.sh
./download_model.sh
 
