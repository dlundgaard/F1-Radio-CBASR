#!usr/bin/env/bash

echo "[INFO] running setup"

sudo apt -q update && sudo apt install -y -q ffmpeg
pip install uv
uv venv .venv --seed --allow-existing
source .venv/bin/activate
uv pip install --link-mode=copy -r requirements.txt
python -m ipykernel install --user --name=project_kernel

echo "[SUCCESS] setup completed"