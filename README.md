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
pkg install git termux-api
termux-speech-to-text
```
Grant microphone permission when prompted.

Then Clone the repo:
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
- Clone `whisper.cpp`
- Build whisper.cpp binaries
- Download the Tiny English Whisper model

The initial setup may take several minutes depending on your device performance and internet connection.

Some older devices may take longer during compilation because compiling C++ on a phone is exactly as unreasonable as it sounds.

---

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

## Configuration

You can adjust sensitivity settings inside `main.py`.

Example:

```python
ENERGY_THRESHOLD = 0.03
```
### Lower values
- Increase microphone sensitivity
- Detect quieter voices more easily
- Increase the chance of false triggers from background noise

### Higher values
- Reduce microphone sensitivity
- Filter background noise more effectively
- May ignore softer speech

Recommended range:

```python
0.02 - 0.05
```
A quieter room can use lower values, while noisy environments usually require higher thresholds.

---

## Project Structure

```text
Termux-STT/
├── main.py
├── requirements.txt
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

---

## License

MIT

Do whatever you want, hust don't be evil.
