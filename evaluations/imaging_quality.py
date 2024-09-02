import torch
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
import argparse
import json
import os
import math
import numpy as np
from decord import VideoReader

def load_video(video_path,data_transform=None, return_tensor=True, width=None, height=None):
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frames = video_reader.get_batch([math.floor(len(video_reader)*0.2),math.floor(len(video_reader)*0.5),math.floor(len(video_reader)*0.8)])  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError
    
    frames = buffer
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames

def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)
        if preprocess_mode == 'shorter_centercrop':
            images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

def compute_imaging_quality(args):
    model_path = args.model_path
    model = MUSIQ(pretrained_model_path=model_path)
    model.to('cuda')
    model.training = False
    device='cuda'

    video_list = get_video_list(args)
    preprocess_mode = args.mode
    video_results = []
    for video in tqdm(video_list):
        try:
            video_path=args.vid_dir+'/'+video['basic']['clip_path']
            images = load_video(video_path)
            images = transform(images, preprocess_mode)
            acc_score_video = 0.
            for i in range(len(images)):
                frame = images[i].unsqueeze(0).to(device)
                score = model(frame)
                acc_score_video += float(score)
            video['scene']['imaging_quality']=acc_score_video/len(images)
            with open(args.out_dir+'/'+video['basic']['clip_id']+'.json','w') as file:
                json.dump(video, file, indent=4)
        except Exception as e:
            print("An error occurred:", str(e))
            continue

def get_video_list(args):
    num=args.num_process
    no=args.mp_no
    metadata_path =args.metadata_path
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
        
    
    length=len(metadata_list)
    if no!=num-1:
        metadata_list=metadata_list[length//num*no:length//num*(no+1)]
    else:
        metadata_list=metadata_list[length//num*no:]
    save_path=args.out_dir
    metadata_list=[data for data in metadata_list if not os.path.exists(save_path+'/'+data['basic']['clip_id']+'.json')]
    
    return metadata_list

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", type=str, default="")
    parser.add_argument("--metadata_path",type=str,default="")
    parser.add_argument('--model_path', default="./models/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth", type=str, help='')
    parser.add_argument("--out_dir",type=str,default="./test")
    parser.add_argument('--mp_no',default=0,type=int)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument('--mode',default='shorter',type=str)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    compute_imaging_quality(args)
