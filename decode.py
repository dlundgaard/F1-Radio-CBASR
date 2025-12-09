import sys, os
import re
import time
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import pprint

import torch
import whisper
import editdistance
from dataloader import get_dataloader, BiasingProcessor
from whisper.model import WhisperBiasing
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from whisper.normalizers.english import EnglishTextNormalizer

parser = argparse.ArgumentParser(description="Whisper Contextual Biasing")

os.makedirs("exports/", exist_ok=True)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--test_json", type=str, default="data/transcriptions_with_context.json")
parser.add_argument("--biasinglist", type=str, default="data/biasing_list.txt")
parser.add_argument("--biasing", action="store_true")
parser.add_argument("--deepbiasing", action="store_true")
parser.add_argument("--modeltype", type=str, default="base.en")
parser.add_argument("--beamsize", type=int, default=3)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--expdir", type=str, default="exports/")
parser.add_argument("--loadfrom", type=str, default="")
parser.add_argument("--use_gpt2", action="store_true")
parser.add_argument("--save_nbest", action="store_true")
parser.add_argument("--lm_weight", type=float, default=0)
parser.add_argument("--attndim", type=int, default=256)
parser.add_argument("--maxKBlen", type=int, default=10)
parser.add_argument("--dropentry", type=float, default=0.0)
parser.add_argument("--normalise", action="store_true")
parser.add_argument("--logfile", type=str, default="log")

args = parser.parse_args()

def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, "a+") as f_log:
            f_log.write(s + "\n")

shallowfusion = args.use_gpt2
useGPT = None
GPTtokenizer = None
normaliser = EnglishTextNormalizer()
logfile = args.logfile if args.logfile != "" else os.path.join(args.expdir, "log.txt")
if args.use_gpt2:
    GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(args.device)
    GPThiddim = GPTmodel.config.n_embd
else:
    GPTmodel = None

if args.loadfrom != "":
    biasing_model = torch.load(args.loadfrom)
    biasing_model.eval()
    model = biasing_model.whisper
    useGPT = getattr(biasing_model, "useGPT", False)
    if useGPT or args.use_gpt2:
        GPTtokenizer = GPT2Tokenizer.from_pretrained("gpt2")
else:
    model = whisper.load_model(args.modeltype).eval()
    biasing_model = None
    useGPT = False

shallowfusion = args.use_gpt2
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")

####################
# Data Loader
####################
testloader = get_dataloader(
    args.test_json,
    args.eval_batch_size,
    loadtarget=False,
    tokenizer=tokenizer,
    biasing=args.biasing,
    shuffle=False,
)
biasproc = BiasingProcessor(tokenizer, args.biasinglist, ndistractors=args.maxKBlen, drop=args.dropentry)

import dataclasses

print("Decoding with" + "\n" + pprint.pformat(vars(args)))
start = time.time()
for idx, data in tqdm(list(enumerate(testloader)), smoothing=0):
    identifiers, audio_features, target_transcriptions, blist = data
    audio_features = audio_features.to(model.device)
    origtree = biasproc.get_lextree(blist)

    if biasing_model is not None and getattr(biasing_model, "GNN", None) is not None:
        biasing_model.GNN(origtree, model.decoder.token_embedding)

    options = whisper.DecodingOptions(
        language="en",
        fp16=False,
        without_timestamps=True,
        biasing=args.biasing,
        biasingmodule=biasing_model,
        origtree=origtree,
        shallowfusion=shallowfusion,
        useGPT=useGPT,
        GPT2=GPTmodel,
        lm_weight=args.lm_weight,
        GPT2tokenizer=GPTtokenizer,
        beam_size=args.beamsize,
    )
    try:
        batch_results = whisper.decode(model, audio_features, options)
        with open("data/transcriptions.tsv", mode="a") as file:
            for identifier, result in zip(identifiers, batch_results):
                # print(f"{identifier}: {result.text}")
                # print(dataclasses.asdict(result))
                file.write("\t".join((
                    str(datetime.now()),
                    identifier,
                    args.modeltype,
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

    # print(str(identifier))

