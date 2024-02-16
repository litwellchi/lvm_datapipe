import time
import argparse
import random
import torch
import numpy as np
import os 
import re
import json
import transformers
from torch.utils.data.distributed import DistributedSampler
from MACVDataset import MACCaptionDataset
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
    parser.add_argument('--hf_key', default='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', help='hf access tokens for loading llama')
    parser.add_argument('--metadata_path', default='configs/meta_config.yaml', help='metadata file name. Please keep in form of metadata_{}.json')
    parser.add_argument('--seed', default=12, help='random seed')
    parser.add_argument('--batch_size', default=8, type=int, help='inference batch size')
    parser.add_argument('--llama_path', default='meta-llama/Llama-2-13b-chat-hf', help='Path to the llama weight file, or just llama hard code of huggingface')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='devices')
    parser.add_argument('--local-rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs for distributed training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of cpu workers for dataloader')
    args = parser.parse_args()
    return args

def format_prompt(captions):
    prefix_prompt = """"
    Summarize nouns in the following setences in a python list format like ["","",""] of single words.
    Here are the setences: ###
    """
    prompt = prefix_prompt+str(captions)+"### Answer:"
    return prompt

def get_wordlist(strs):
    match = re.search(r'Answer:\s*\[(.*?)\]', strs)
    if match:
        answer = match.group(1)
        answer = answer.replace("\"","").replace(" ","").split(",")
        print(answer)
    return answer

def main(args):
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Dataloader

    dataset = MACCaptionDataset(args.metadata_path)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    start_time = time.time()


    #Load model
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_path,use_auth_token=args.hf_key)
    model = LlamaForCausalLM.from_pretrained(args.llama_path,use_auth_token=args.hf_key)
    pipeline = transformers.pipeline("text-generation", 
                                    model=model,
                                    tokenizer=tokenizer,                                 
                                    torch_dtype=torch.float16, 
                                    device = torch.device('cuda', index=0)
                                    )
    
    # Inference 

    for batch_frame, idx in tqdm.tqdm(dataloader):
        pass

    captions ="a light blue background with squares in the middle of it . a blue background with squares and squares and squares with the words  it is strongly believed to enforce rape written on top of it . a person standing in front of a fence with a text ."
    # captions ="a chain hanging from a metal chain next to a blue blanket . a chain hanging from a metal object . a close - up of a chain hanging on a wall ."
    # model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    prompt = format_prompt(captions)
    s_t = time.time()
    sequences = pipeline([prompt], 
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        max_length=128,
                        truncation=True)

    for sequence in sequences:
        match = get_wordlist(sequence[0]['generated_text'])
        print(f"chatbot: \n {match}")
    print(f"using time {time.time()-s_t}s")

if __name__ == "__main__":
    args = get_parser()
    main(args)
    
