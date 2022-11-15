import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    # load model
    #model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # original example
    batchsize = 1
    seqlen = 4096

    sequence = torch.rand([batchsize,seqlen, 768])
    #sequence = torch.rand([batchsize,seqlen, 1024])
    # run inference with checkpoint on dummy sequence of desired length (do this N times)
    #model.bert.encoder.forward(sequence)

    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    #sequence = torch.ones([seqlen, 768])
    print(model.bert.encoder)
    N = 12*100
    model = model.to(device)
    
#    with torch.profiler.profile(
#    activities=[
#        torch.profiler.ProfilerActivity.CPU,
#        torch.profiler.ProfilerActivity.CUDA,
#    ]
#    ) as p:
#        for i in range(0,N):
#    outputs = model.bert.encoder(sequence.to(device)) 
#    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))           
    outputs = model.bert.encoder(sequence.to(device), gpu_profile=True, num_iters=N) 

if __name__ == "__main__":
    main()
