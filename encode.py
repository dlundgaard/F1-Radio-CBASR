import json
import torch
import argparse
import whisper
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Whisper Contextual Biasing")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--transcribed_json", type=str, default="data/transcribed_with_reference.json")
parser.add_argument("--modeltype", type=str, default="base.en")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

transcribed = pd.read_json(args.transcribed_json)

features = dict()
for index, example in tqdm(
    list(transcribed.iterrows()),
    desc="Caching audio spectrogram features",
):
    utterance = " " + example["human_transcription"].upper()
    audio = whisper.pad_or_trim(whisper.load_audio(example["file_path"]))

    log_mel_spectra = whisper.log_mel_spectrogram(audio, n_mels=128 if args.modeltype == "turbo" else 80)
    feature_dump_filepath = f"data/audio_features/{example['identifier']}.pt"
    torch.save(log_mel_spectra, feature_dump_filepath)
    features[example["identifier"]] = dict(
        fbank=feature_dump_filepath,
        words=utterance,
    )

with open("data/biasing_list.txt") as file:
    biasing_terms = [word.strip() for word in file]

for utterance_id, utterance in features.items():
    biasing_term_occurences = []
    words = [word.strip(r"\"!#$%&'()*+,./:;<=>?@[\]^_`{|}~") for word in utterance["words"].split()]
    features[utterance_id]["blist"] = list(set([word for word in words if word in biasing_terms]))

with open("data/transcriptions_with_context.json", mode="w") as file:
    json.dump(features, file, indent=4)