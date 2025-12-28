import os
import re
import json
import torch
import argparse
import whisper
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Whisper Contextual Biasing")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--transcribed_json", type=str, default="data/with_human_reference.json")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

transcribed = pd.read_json(args.transcribed_json)
with open("data/biasing_list.txt") as file:
    biasing_terms = [word.upper().strip() for word in file if word]

seen_biasing_terms = set()

examples = dict()
for index, example in tqdm(
    list(transcribed.iterrows()),
    desc="Building training and evaluation dataset",
):
    clip_identifier = example["identifier"]

    log_mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(example["file_path"])), n_mels=80)
    feature_dump_filepath = os.path.join("data", "audio_features", example["identifier"] + ".pt")
    torch.save(log_mel, feature_dump_filepath)

    utterance = re.sub(r"[^A-Za-z0-9 *]", "", example["human_transcription"]).strip().upper()
    utterance_biasing_terms = set([word for word in utterance.split() if word in biasing_terms])
    seen_biasing_terms.update(utterance_biasing_terms)

    examples[clip_identifier] = dict(
        fbank=feature_dump_filepath,
        words=" " + utterance + " ",
        blist=list(utterance_biasing_terms),
    )

with open("data/benchmarking_examples.json", mode="w") as file:
    json.dump(examples, file, indent=4)

with open("data/training_examples.json", mode="w") as file:
    json.dump({clip_identifier: entry for clip_identifier, entry in examples.items() if len(entry["blist"])}, file, indent=4)

with open("data/biasing_list_seen.txt", mode="w") as file:
    file.write("\n".join(sorted(seen_biasing_terms)))
