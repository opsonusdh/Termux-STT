# Termux-STT

Offline real-time Speech-to-Text for Termux using `whisper.cpp`.

## Features

- Real-time microphone transcription
- Voice-activated recording
- Offline speech recognition
- Low latency
- Lightweight setup
- Runs fully inside Termux

---

## Requirements

- Android device
- Termux
- Termux:API app
- Internet connection for initial setup only

---

## Installation

Install the App Termux and Termux-api:
[Termux](https://f-droid.org/en/packages/com.termux/)
[Termux-api](https://f-droid.org/en/packages/com.termux.api/)

Then install the package inside Termux:
```bash
pkg install git python termux-api
```
Clone the repo:
```bash
git clone https://github.com/opsonusdh/Termux-STT/
cd Termux-STT/
```

---

## Setup

Run:

```bash
bash setup.sh
```
This script will:

- Install required Termux packages
- Install Python dependencies
- Ask microphone permission
- Clone `whisper.cpp`
- Build whisper.cpp binaries
- Download the Whisper model of your choice

The initial setup may take several minutes depending on your device performance and internet connection.

Some older devices may take longer during compilation because compiling C++ on a phone is exactly as unreasonable as it sounds.

---

## Groq support
If you want faster transcription with high precision, you can use groq.
just do:
```bash
nano api.keys
```
here add all groq API keys you have:
```text
gsk_
gsk_
...
```
## Run

Start the application:

```bash
python main.py
```
or, as a module:
```python
from main import listen

for text in listen(once=False):
    print("User said:")
    print(text)
```

The program will continuously listen for speech using your device microphone.

When speech is detected:
1. Audio recording starts automatically
2. Speech audio is buffered temporarily
3. Silence stops the recording
4. The recorded audio is transcribed locally using `whisper.cpp`

The entire pipeline runs offline after setup.

---

## Project Structure

```text
Termux-STT/
├── main.py
├── requirements.txt
├── download_model.sh
└── setup.sh
```

---

## Notes

- Fully offline after installation
- Optimized for low latency
- Tiny model prioritizes speed and low RAM usage
- Works best in quieter environments

Sounds such as keyboards, fans, traffic, birds, or other environmental noise may occasionally trigger speech detection depending on your threshold settings. Tiny models are surprisingly capable, but not psychic. Yet.

---

## Future Improvements

- Proper Voice Activity Detection (VAD)
- Wake-word activation
- Streaming transcription
- Multi-language support
- Live subtitle mode
- Terminal overlay interface

---

## Credits

- whisper.cpp
  https://github.com/ggerganov/whisper.cpp

- OpenAI Whisper

- Groq

---

## License

MIT

Do whatever you want, just don't be evil.

---

## Final note

- It is intended to integrate with other Termux projects, including Termux-TUI.
- Improvements, suggestions, and testing are welcome.
