import torch
import torch.nn as nn
import math
from transformers import GPT2Config
from transformers.activations import gelu_new

GPT2_CONFIG_PATH = './static_files/config.json'
GPT2_WEIGHTS_PATH = './static_files/gpt_pytorch_model.bin'


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.config = GPT2Config.from_pretrained(GPT2_CONFIG_PATH)
        self.model_weight = torch.load(GPT2_WEIGHTS_PATH, map_location='cpu')
        self.load_weights()
    def forward(self, x):
        return x

    def load_weights(self):
        pass
        