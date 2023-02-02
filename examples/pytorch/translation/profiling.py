import logging
import os
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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

def main():
    # load model
    model = Seq2SeqTrainer.from_pretrained("mt5-small", ignore_mismatched_sizes=True)

    seqlen = 128 + 1 # 128, 256, 512, 1024, 2048, 4096
    sequence = torch.LongTensor([[1]])
    #device = "cuda:6" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    N = 10
    model = model.to(device)
    for i in range(0,N):
#    	outputs = model.transformer(sequence.to(device))
        outputs = model.generate(input_ids=sequence, max_length=seqlen)

if __name__ == "__main__":
    main()
