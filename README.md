# Transcribing Formula 1 Team Radio with TCPGen Contextual Biasing 🏎️📻

![](assets/banner_figure.png)

## Dependencies

System dependencies: `Python 3.8+` and `ffmpeg`.

Python package dependencies: [`requirements.txt`](requirements.txt)

## Running the Experiment

Executing the [`setup.sh`](setup.sh) shell script will install the `ffmpeg` system dependency and use [`uv`](https://github.com/astral-sh/uv) to set up a virtual environment which satisfies all Python package dependencies.

Running [`run.sh`](run.sh) will carry out an experimental workflow generating data for comparing "stock" Whisper against Whisper with TCPGen.

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

Tested on Ubuntu Linux 24.04.2 with Python 3.12.3, with computation performed on a machine with Xeon Gold 6130 CPU and 96 GB RAM from the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark.

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

## Acknowledgements

The TCPGen contextual biasing Whisper implementation used here is courtesy of [BriansIDP/WhisperBiasing](https://github.com/BriansIDP/WhisperBiasing).