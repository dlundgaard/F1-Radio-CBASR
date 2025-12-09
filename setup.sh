#!usr/bin/env/bash

echo "[INFO] running setup"
echo "[INFO] installing ffmpeg"
sudo apt -q update && sudo apt install -y -q ffmpeg
echo "[INFO] setting up Python virtual enviroment"
pip install uv
uv venv .venv --seed --allow-existing
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=project_kernel
echo "[SUCCESS] setup completed"