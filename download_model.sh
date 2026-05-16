#!/data/data/com.termux/files/usr/bin/bash

cd whisper.cpp || exit 1

# =========================================================
# 24-bit bold colors
# =========================================================

BOLD='\033[1m'
RESET='\033[0m'

# Gradient: green -> yellow -> orange -> red

C_TINY='\033[1;38;2;0;255;120m'
C_TINY_EN='\033[1;38;2;60;255;80m'

C_BASE='\033[1;38;2;170;255;0m'
C_BASE_EN='\033[1;38;2;220;255;0m'

C_SMALL='\033[1;38;2;255;220;0m'
C_SMALL_EN='\033[1;38;2;255;180;0m'

C_MEDIUM='\033[1;38;2;255;120;0m'
C_MEDIUM_EN='\033[1;38;2;255;80;0m'

C_LARGE='\033[1;38;2;255;40;40m'
C_LARGE_V3='\033[1;38;2;255;0;0m'

MODELS_DIR="./models"

echo ""
echo -e "${BOLD}Choose Whisper model quality:${RESET}"
echo ""

echo -e " 0) ${BOLD}Close${RESET}                -> Exit installer"

echo -e " 1) ${C_TINY}tiny${RESET}                -> fastest multilingual"
echo -e " 2) ${C_TINY_EN}tiny.en${RESET}             -> fastest English-only"

echo -e " 3) ${C_BASE}base${RESET}                -> balanced multilingual"
echo -e " 4) ${C_BASE_EN}base.en${RESET}             -> balanced English-only"

echo -e " 5) ${C_SMALL}small${RESET}               -> improved accuracy"
echo -e " 6) ${C_SMALL_EN}small.en${RESET}            -> improved English-only"

echo -e " 7) ${C_MEDIUM}medium${RESET}              -> strong multilingual"
echo -e " 8) ${C_MEDIUM_EN}medium.en${RESET}           -> strong English-only"

echo -e " 9) ${C_LARGE}large${RESET}               -> huge multilingual"
echo -e "10) ${C_LARGE_V3}large-v3${RESET}            -> best quality, thermonuclear"

echo ""

read -p "Enter choice [0-10]: " choice

case $choice in
    0)
        exit 0
        ;;

    1)
        MODEL="tiny"
        ;;

    2)
        MODEL="tiny.en"
        ;;

    3)
        MODEL="base"
        ;;

    4)
        MODEL="base.en"
        ;;

    5)
        MODEL="small"
        ;;

    6)
        MODEL="small.en"
        ;;

    7)
        MODEL="medium"
        ;;

    8)
        MODEL="medium.en"
        ;;

    9)
        MODEL="large"
        ;;

    10)
        MODEL="large-v3"
        ;;

    *)
        echo ""
        echo -e "${C_BASE_EN}Invalid choice. Defaulting to base.en${RESET}"
        MODEL="base.en"
        ;;
esac

echo ""

FOUND_MODELS=$(find "$MODELS_DIR" -maxdepth 1 -type f -name "ggml-*.bin")

if [ -n "$FOUND_MODELS" ]; then
    echo "Existing models found:"
    echo "$FOUND_MODELS"

    echo ""
    echo "Deleting old models..."

    find "$MODELS_DIR" -maxdepth 1 -type f -name "ggml-*.bin" -delete

    echo "Old models deleted."
fi

echo ""
echo "Downloading model: $MODEL"

bash ./models/download-ggml-model.sh "$MODEL"

echo ""
echo "Model installation complete."

cd ..