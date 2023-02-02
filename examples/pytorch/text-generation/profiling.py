
import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


def main():
    # load model
    model = GPT2LMHeadModel.from_pretrained("gpt2", ignore_mismatched_sizes=True)

    seqlen = 4096 + 1 # 128, 256, 512, 1024, 2048, 4096
    sequence = torch.LongTensor([[1]])
    #device = "cuda:6" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    N = 1
    model = model.to(device)
    for i in range(0,N):
#    	outputs = model.transformer(sequence.to(device))
        outputs = model.generate(input_ids=sequence, max_length=seqlen)
        print(outputs.shape)

if __name__ == "__main__":
    main()
