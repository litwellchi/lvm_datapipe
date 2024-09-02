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


class VideoDataset(Dataset):
    def __init__(self, metadata_list, video_path, num_frames, transform):
        self.metadata_list = metadata_list
        self.video_path = video_path
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        clip_path = os.path.join(self.video_path, metadata["basic"]["clip_path"])
        frames = self.getImageFromVideo(clip_path, self.num_frames)
        if frames == None: return None,idx
        trans_frames = [self.transform(frame).unsqueeze(0) for frame in frames]
        batch_frame = torch.cat(trans_frames, dim=0)
        return batch_frame, idx

    def getImageFromVideo(self, clip_path, num_frames):
        try:
            cap = cv2.VideoCapture(clip_path)
            frame_list = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= num_frames:
                query_list = [0,0,0]
            else:
                query_list =  [0, frame_count // 2, frame_count - 1]
            for i in query_list:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = cap.read()
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
    save_metadata_path = metadata_path.replace('metadata', 'metadata_catpion')
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    # TODO
    # rough filtered_list
    metadata_list = [item for item in metadata_list if item['basic']["clip_duration"] > 1.0]
    

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained=args.coca_path,
        device="cuda"
    )

    model = model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    dataset = VideoDataset(metadata_list, args.video_path, args.num_frames, transform)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    start_time = time.time()
    sub_metadata_list=[]
    for batch_frame, idx in tqdm.tqdm(dataloader):
        try:
            batch_frame = batch_frame.view(-1, *batch_frame.shape[2:]).to(args.local_rank)
            with torch.no_grad(), torch.cuda.amp.autocast():
                generated = model.module.generate(batch_frame)

            generated = generated.view(-1, args.num_frames, generated.shape[-1])

            for count in range(idx.shape[0]):
                result_list = [open_clip.decode(generated[count][i]).split("<end_of_text>")[0].replace("<start_of_text>", "") for i in
                               range(args.num_frames)]
                print(idx,count,idx[count],metadata_list[idx[count]])
                metadata_list[idx[count]]['misc']['frame_caption'] = result_list
                sub_metadata_list.append(metadata_list[idx[count]])
        except Exception as e:
            print("An error occurred:", str(e))
            continue

    # 
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
    parser.add_argument('--video_path', default='freeguy_test', help='Path to the video folder')
    parser.add_argument('--metadata_path', default='metadata.json', help='metadata file name. Please keep in form of metadata_{}.json')
    parser.add_argument('--num_frames', default=3, help='number of frames extract from one clip video')
    parser.add_argument('--batch_size', default=8, type=int, help='inference batch size')
    parser.add_argument('--coca_path', default='coca/open_clip_pytorch_model.bin', help='Path to the coca weight file')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='devices')
    parser.add_argument('--local-rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs for distributed training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of cpu workers for dataloader')
    args = parser.parse_args()

    main(args)

