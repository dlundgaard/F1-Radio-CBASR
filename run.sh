#!usr/bin/env/bash

echo "[INFO] running experiment workflow"

source .venv/bin/activate

python3 encode.py

python3 train.py --runidentifier "TCPGenWhisper"

python3 decode.py --modelcheckpoint="TCPGenWhisper"
python3 decode.py

echo "[SUCCESS] experiment workflow completed"