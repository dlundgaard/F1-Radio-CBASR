import json
import re
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

features = dict()
for index, example in tqdm(
    list(transcribed.iterrows()),
    desc="Extracting audio spectrogram features",
):
    utterance = " " + example["human_transcription"].upper().strip() + " "
    audio = whisper.pad_or_trim(whisper.load_audio(example["file_path"]))

    log_mel_spectra = whisper.log_mel_spectrogram(audio, n_mels=80)
    feature_dump_filepath = f"""data/audio_features/{example["identifier"]}.pt"""
    torch.save(log_mel_spectra, feature_dump_filepath)

    utterance_words = re.sub(r"[^A-Za-z0-9 *]", "", utterance).split()
    utterance_biasing_terms = list(set([word for word in utterance_words if word in biasing_terms]))
    seen_biasing_terms.update(utterance_biasing_terms)

    features[example["identifier"]] = dict(
        fbank=feature_dump_filepath,
        words=utterance,
        blist=utterance_biasing_terms,
    )

with open("data/transcriptions_with_context.json", mode="w") as file:
    json.dump(features, file, indent=4)

with open("data/biasing_list_seen.txt", mode="w") as file:
    file.write("\n".join(sorted(seen_biasing_terms)))
