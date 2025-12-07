#!usr/bin/env/bash

echo "[INFO] running setup"
sudo apt update -qqq && sudo apt install -y -qqq ffmpeg
pip install uv
uv venv .venv --seed --allow-existing
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=project_kernel
echo "[SUCCESS] setup completed"
