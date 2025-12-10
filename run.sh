#!usr/bin/env/bash

echo "[INFO] running experiment workflow"
source .venv/bin/activate

python3 encode.py --modeltype="base.en"
python3 decode.py --modeltype="base.en"
python3 train.py --modeltype="base.en" --runidentifier "TCPGen"
python3 train.py --modeltype="base.en" --useGPT --runidentifier "TCPGen+GPT2"
python3 decode.py --modeltype="base.en" --modelcheckpoint="base.en_TCPGen.best.pt"
python3 decode.py --modeltype="base.en" --useGPT --modelcheckpoint="base.en_TCPGen+GPT2.best.pt"

echo "[SUCCESS] experiment workflow completed"