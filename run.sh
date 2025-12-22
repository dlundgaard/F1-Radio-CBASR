#!usr/bin/env/bash

echo "[INFO] running experiment workflow"

source .venv/bin/activate

python3 encode.py

python3 train.py --modeltype="base.en" --runidentifier "TCPGenWhisper"

python3 decode.py --modeltype="base.en" --modelcheckpoint="TCPGenWhisper"
python3 decode.py --modeltype="base.en"

echo "[SUCCESS] experiment workflow completed"