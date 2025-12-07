import json
import torch
import whisper
import pandas as pd
from tqdm import tqdm

transcribed = pd.read_json("data/transcribed_with_reference.json")

features = dict()
for index, example in tqdm(
    list(transcribed.iterrows()),
    desc="Caching audio spectrogram features",
):
    file_name = example["file_path"].split("/")[1]
    utterance = " " + example["human_transcription"].upper()
    audio = whisper.pad_or_trim(whisper.load_audio("data/" + example["file_path"]))

    log_mel_cached = whisper.log_mel_spectrogram(audio, n_mels=128)
    feature_dump_filepath = "data/audio_features/" + file_name + ".pt"
    torch.save(log_mel_cached, feature_dump_filepath)
    features[file_name] = {"fbank": feature_dump_filepath, "words": utterance}

with open("data/biasing_list.txt") as file:
    biasing_terms = [word.strip() for word in file]

for utterance_id, utterance in features.items():
    biasing_term_occurences = []
    words = [word.strip(r"\"!#$%&'()*+,./:;<=>?@[\]^_`{|}~") for word in utterance["words"].split()]
    features[utterance_id]["blist"] = list(set([word for word in words if word in biasing_terms]))

with open("data/transcriptions_with_context.json", mode="w") as file_out:
    json.dump(features, file_out, indent=4)