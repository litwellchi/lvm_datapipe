import time
import argparse
import random
import torch
import numpy as np
import os 
import re
import json
import csv
import transformers
from torch.utils.data.distributed import DistributedSampler
from data_schema.macvidataset import MACCaptionDataset
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
    parser.add_argument('--hf_key', default='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', help='hf access tokens for loading llama')
    parser.add_argument('--metadata_path', default='configs/meta_config.yaml', help='metadata file name. Please keep in form of metadata_{}.json')
    parser.add_argument('--csv_path', default='sum_caption.csv', help='metadata file name. Please keep in form of metadata_{}.json')
    parser.add_argument('--seed', default=12, help='random seed')
    parser.add_argument('--batch_size', default=32, type=int, help='inference batch size')
    parser.add_argument('--llama_path', default='meta-llama/Llama-2-13b-chat-hf', help='Path to the llama weight file, or just llama hard code of huggingface')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='devices')
    parser.add_argument('--local-rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Number of GPUs for distributed training')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of cpu workers for dataloader')
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
    try:
        match = re.search(r'Answer:\s*\[(.*?)\]', strs)
        if match:
            answer = match.group(1)
            answer = answer.replace("\"","").replace(" ","").split(",")
        return answer
    except:
        return []

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    print(len(batch))
    if len(batch) == 0:
        return None,0
    return torch.utils.data.dataloader.default_collate(batch)

def main(args):
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Dataloader

    dataset = MACCaptionDataset(args.metadata_path)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,)
    start_time = time.time()


    #Load model
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_path,use_auth_token=args.hf_key)
    model = LlamaForCausalLM.from_pretrained(args.llama_path,use_auth_token=args.hf_key)
    pipeline = transformers.pipeline("text-generation", 
                                    model=model,
                                    tokenizer=tokenizer,                                 
                                    torch_dtype=torch.int8, 
                                    device = torch.device('cuda', index=0)
                                    )
    
    write_lines = []
    # Inference 
    for caption,path,idx in tqdm.tqdm(dataloader):
        prompts = [format_prompt(c) for c in caption]
        s_t = time.time()
        sequences = pipeline(prompts, 
                            do_sample=True,
                            top_k=10,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            max_length=156,
                            truncation=True)
        for i in range(args.batch_size):
            match = get_wordlist(sequences[i][0]['generated_text'])
            write_to_csv(args.csv_path,[path[i],match])
            # write_lines.append([path,match])
        break
    
    # saving to the csv(for multi prompt)
    print(f"using time {time.time()-s_t}s")

if __name__ == "__main__":
    args = get_parser()
    main(args)
    
