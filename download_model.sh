#!/data/data/com.termux/files/usr/bin/bash

cd whisper.cpp || exit 1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
ORANGE='\033[38;5;208m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "Choose Whisper model quality:"
echo -e "0) Close   -> Close the model Download system"
echo -e "1) ${GREEN}tiny${NC}   -> fastest, lowest accuracy"
echo -e "2) ${YELLOW}base${NC}   -> balanced"
echo -e "3) ${ORANGE}medium${NC} -> slower, much better accuracy"
echo -e "4) ${RED}large${NC}  -> best quality, phone may suffer emotionally"
echo ""

read -p "Enter choice [0-4]: " choice

case $choice in
    0)
        exit 0;
        ;;
    1)
        MODEL="tiny.en"
        ;;
    2)
        MODEL="base.en"
        ;;
    3)
        MODEL="medium.en"
        ;;
    4)
        MODEL="large-v3"
        ;;
    *)
        echo "Invalid choice. Defaulting to base.en"
        MODEL="base.en"
        ;;
esac

echo ""
echo "Downloading model: $MODEL"

bash ./models/download-ggml-model.sh "$MODEL"

cd ..


