# Transcribing Formula 1 Radio Message with TCPGen Contextual Biasing 🏎️🎙️📻

## About

Modern end-to-end automatic speech recognition (ASR) systems are hugely adept at transcribing human speech to text. However, this is true primarily for everyday generic speech similar to what the system was exposed to during pretrained. 
When these systems are presented with speech containing very domain-specific language they will tend to get tripped up and stumble. When they encounter words/terms that differ markedly from everyday speech – e.g., by including rare or novel words – these systems can fail spectacularly because they are preconditioned to assume that any utterance shares its vocabulary with the speech it was exposed to during its (pre)training.

## Dependencies

System dependencies: `Python 3.8+` and `ffmpeg`
Python package dependencies: [`requirements.txt`](requirements.txt)

Running the [`setup.sh`](setup.sh) shell script will install `ffmpeg` and use [`uv`](https://github.com/astral-sh/uv) to set up a Python virtual environment satisfying all package dependencies.

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.