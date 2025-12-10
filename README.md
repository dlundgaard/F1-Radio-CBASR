# Transcribing Formula 1 Radio Message with TCPGen Contextual Biasing 🏎️🎙️📻

Modern end-to-end automatic speech recognition (ASR) systems are exceptionally adept at transcribing everyday generic human speech to text, but these systems can get tripped up when presented with speech comprising rare, domain-specific language. 
When encountering utterances that differ markedly from everyday speech by including rare or novel words – so-called out-of-distribution examples – these systems may fail spectacularly because they are so heavily preconditioned to consider any utterance they as an instance of the speech vocabulary it was exposed to during (pre)training.

## Dependencies

System dependencies: `Python 3.8+` and `ffmpeg`
Python package dependencies: [`requirements.txt`](requirements.txt)

## Steps to reproduce

Running the [`setup.sh`](setup.sh) shell script will install `ffmpeg` and use [`uv`](https://github.com/astral-sh/uv) to set up a Python virtual environment satisfying all package dependencies.

Running [`run.sh`](run.sh) will carry out an experimental workflow generating data for comparing "stock" Whisper against Whisper with TCPGen and Whisper with TCPgen and GPT2.

```bash
$ source setup.sh
[INFO] running setup
...
[SUCCESS] setup completed

$ source run.sh
[INFO] running experiment workflow
...
[SUCCESS] experiment workflow completed
```

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
