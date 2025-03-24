from transformers import GPT2Tokenizer
import time
import matplotlib.pyplot as plt
from myGPT2Model import * # model builder file
import numpy as np
import torch

GPT2_CONFIG_PATH = './static_files/config.json'
config = GPT2Config.from_pretrained(GPT2_CONFIG_PATH)

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode("life is a long journey, if you keep going", return_tensors="pt")

    print(input_ids)

    model = GPT2Model(config)
    infer_start_time = time.time()
    infer_per_token_time = []
    new_tokens_number = 1

    with torch.no_grad():
        for i in range(new_tokens_number):
            infer_token_start_time = time.time()
            result = model(input_ids)
            infer_per_token_time.append((time.time() - infer_token_start_time))
            print('token id: ', result) 
            new_ids = torch.tensor(result)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            decoded_string = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print('Decoded string:\n*********\n', decoded_string, '\n***********\n')

    infer_end_time = time.time()
    print('time taken for inference: ', infer_end_time - infer_start_time)
    print("time for each token: ", (infer_end_time - infer_start_time)/new_tokens_number)

    x = np.arange(len(infer_per_token_time))
    plt.plot(x, infer_per_token_time)
    plt.savefig('./static_files/inference_time.png')
    

main()