import os
import time
import json
import cv2
import torch
import argparse
import tqdm
from PIL import Image
import open_clip
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.llava.conversation import conv_templates, SeparatorStyle
from llava.llava.model.builder import load_pretrained_model
from llava.llava.utils import disable_torch_init
from llava.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

class VideoDataset(Dataset):
    def __init__(self, metadata_list, video_path, num_frames, transform,image_processor=None,model_config=None):
        self.metadata_list = metadata_list
        self.video_path = video_path
        self.num_frames = num_frames
        self.transform = transform
        self.image_processor = image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        clip_path = os.path.join(self.video_path, f'video_dataset_85/{metadata["basic"]["clip_path"]}')
        frames = self.getImageFromVideo(clip_path, points=[0.2,0.5,0.8])
        if frames == None: return None,idx
        trans_frames = self.transform(frames,self.image_processor,self.model_config)
        batch_frame = trans_frames
        return batch_frame, idx

    def getImageFromVideo(self, clip_path,points=[0.2, 0.5, 0.8]):
        try:
            cap = cv2.VideoCapture(clip_path)
            frame_list = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            query_list = [int(frame_count * point) for point in points]
            for i in query_list:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = cap.read()
                frame_list.append(Image.fromarray(frame).convert("RGB"))
            return frame_list
        except:
            Exception(f"Failed to open video file {clip_path}.")
            return None

def collate_fn(batch):
    batch = [data for data in batch if data[0] is not None]
    if len(batch) == 0:
        return None,0
    return torch.utils.data.dataloader.default_collate(batch)

def main(args):
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    
    metadata_path = os.path.join(args.video_path, args.metadata_path)
    # save_metadata_path = metadata_path.replace('metadata', 'metadata_catpion')
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    # TODO
    # rough filtered_list
    metadata_list = [item for item in metadata_list if item['basic']["clip_duration"] > 1.0]

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    model = model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    dataset = VideoDataset(metadata_list, args.video_path, args.num_frames,process_images,image_processor=image_processor,model_config=model.module.config)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    start_time = time.time()
    sub_metadata_list=[]
    for batch_frame, idx in tqdm.tqdm(dataloader):
        # try:
        image = batch_frame.view(-1, *batch_frame.shape[2:]).to(args.local_rank)
        # print(batch_frame.shape)
        image_size = batch_frame.shape[-2:]
        inp = "please describe this image"
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        if image is not None:
            # first message
            if model.module.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.local_rank)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = model.module.generate(
                input_ids,
                images=image,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)
            outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs
                

                # metadata_list[idx[count]]['misc']['frame_caption'] = result_list
                # sub_metadata_list.append(metadata_list[idx[count]])
        # except Exception as e:
        #     print("An error occurred:", str(e))
        #     continue
        return

    save_metadata_path = metadata_path.replace('metadata', f'metadata_catpion_{args.local_rank}')
    with open(save_metadata_path, 'w') as f:
        json.dump(sub_metadata_list, f)

    dist.barrier()
    if args.local_rank == 0:
        save_metadata_path = metadata_path.replace('metadata', f'metadata_catpion')
        all_caption = []
        for i in range(args.world_size):
          with open(save_metadata_path.replace('metadata_catpion',f"metadata_catpion_{i}"), 'r') as f:
              metadata_list = json.load(f)
              all_caption.extend(metadata_list)
        with open(save_metadata_path, 'w') as f:
          json.dump(all_caption, f)
        print(f"processing time:{time.time()-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
    parser.add_argument('--video_path', default='', help='Path to the video folder')
    parser.add_argument('--metadata_path', default='metadata_caption.json', help='metadata file name. Please keep in form of metadata_{}.json')
    parser.add_argument('--num_frames', default=3, help='number of frames extract from one clip video')
    parser.add_argument('--batch_size', default=8, type=int, help='inference batch size')
    parser.add_argument('--local-rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--world_size', default=6, type=int, help='Number of GPUs for distributed training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of cpu workers for dataloader')
    parser.add_argument('--gpu_ids', default='0,1,2,3,5,6,', help='devices')

    parser.add_argument("--model-path", type=str, default="llava-v1.6-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)

