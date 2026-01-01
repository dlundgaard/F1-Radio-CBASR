import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import pprint

import torch
import whisper
from dataloader import get_dataloader, BiasingProcessor

parser = argparse.ArgumentParser(description="Whisper Contextual Biasing")

os.makedirs("exports/", exist_ok=True)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--test_json", type=str, default="data/benchmarking_examples.json")
parser.add_argument("--biasinglist", type=str, default="data/biasing_list.txt")
parser.add_argument("--modelcheckpoint", type=str, default="stockWhisper")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--beamsize", type=int, default=10)
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--attndim", type=int, default=256)
parser.add_argument("--maxKBlen", type=int, default=100)
parser.add_argument("--dropentry", type=float, default=0)
parser.add_argument("--expdir", type=str, default="exports/")

args = parser.parse_args()

if args.modelcheckpoint != "stockWhisper":
    biasing_model = torch.load(
        os.path.join(args.expdir, args.modelcheckpoint + ".pt"), 
        weights_only=False,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    biasing_model.eval()
    model = biasing_model.whisper
else:
    model = whisper.load_model("base.en").eval()
    biasing_model = None

tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")

####################
# Data Loader
####################
testloader = get_dataloader(
    args.test_json,
    args.eval_batch_size,
    loadtarget=False,
    tokenizer=tokenizer,
    biasing=biasing_model is not None,
    shuffle=False,
)
biasproc = BiasingProcessor(tokenizer, args.biasinglist, ndistractors=args.maxKBlen, drop=args.dropentry)

print("Decoding with" + "\n" + pprint.pformat(vars(args)))
start = time.time()
for idx, data in tqdm(
    list(enumerate(testloader)), 
    smoothing=0,
    desc="Transcribing audio clips",
):
    identifiers, audio_features, target_transcriptions, blist = data
    audio_features = audio_features.to(model.device)
    origtree = biasproc.get_lextree(blist)

    options = whisper.DecodingOptions(
        language="en",
        fp16=False,
        without_timestamps=True,
        biasing=biasing_model is not None,
        biasingmodule=biasing_model,
        origtree=origtree,
        beam_size=args.beamsize,
    )
    try:
        batch_results = whisper.decode(model, audio_features, options)
        with open(f"exports/transcriptions.tsv", mode="a") as file:
            for identifier, result in zip(identifiers, batch_results):
                file.write("\t".join((
                    str(datetime.now()),
                    identifier,
                    args.modelcheckpoint,
                    *[str(getattr(result, field)) for field in (
                        "text",
                        "avg_logprob",
                        "text_nbest",
                        "no_speech_prob",
                        "temperature",
                        "compression_ratio",
                        "sum_logprob_nbest",
                        "token_nbest",
                    )],
                )) + "\n")
    except Exception as exc:
        print(f"[ERROR] {identifier}: {repr(exc)}")