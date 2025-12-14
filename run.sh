#!usr/bin/env/bash

echo "[INFO] running experiment workflow"
source .venv/bin/activate

python3 encode.py
python3 decode.py --modeltype="base.en"
python3 train.py --modeltype="base.en" --runidentifier "TCPGen" --nepochs 10
python3 decode.py --modeltype="base.en" --modelcheckpoint="base.en_TCPGen.best.pt" --biasing
# python3 train.py --modeltype="base.en" --runidentifier "TCPGen+GPT2" --useGPT --nepochs 10
# python3 decode.py --modeltype="base.en" --modelcheckpoint="base.en_TCPGen+GPT2.best.pt" --biasing --useGPT

echo "[SUCCESS] experiment workflow completed"