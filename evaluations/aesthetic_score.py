import os
import time
import json
import cv2
import torch
import argparse
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F

import clip
import time


from PIL import Image
import cv2

#####  This script will predict the aesthetic score for this image file:
start_time = time.time()

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class VideoDataset(Dataset):
    def __init__(self, metadata_list, video_path, num_frames,transform):
        self.metadata_list = metadata_list
        self.video_path = video_path
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        # TODO config the datapath
        clip_path = os.path.join(self.video_path, f'{metadata["basic"]["clip_path"]}')
        frames = self.getImageFromVideo(clip_path, points=[0.2,0.5,0.8])
        if frames == None: return None,idx
        batch_frame = []
        for frame in frames:
            image = self.transform(frame).unsqueeze(0)
            batch_frame.append(image)
        
        return batch_frame, idx,metadata["basic"]["clip_path"]

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
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load("./model/improved-aesthetic-predictor/ava+logos-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=args.local_rank)  #RN50x64   
    model = model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model2 = DDP(model2, device_ids=[args.local_rank], output_device=args.local_rank)
    dataset = VideoDataset(metadata_list, args.video_path, args.num_frames,preprocess)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    start_time = time.time()
    sub_metadata_list=[]
    for batch_frame, idx,batch_path in tqdm.tqdm(dataloader):
        # batch_frame为一个list，len为从每个视频中剪辑的图片，每个元素为一个tensor，size为[bsz,1,3,224,224]
        try:
            with torch.no_grad():
                frames = []
                for image in batch_frame:
                    image_features = model2.module.encode_image(image.squeeze(1).to(args.local_rank))
                    im_emb_arr = normalized(image_features.cpu().detach().numpy())
                    frames.append(im_emb_arr)
                # frames的纬度为[3,bsz,768]
                frames = torch.tensor(np.array(frames)).to(args.local_rank)
                prediction = model(frames.type(torch.cuda.FloatTensor))
                # 将[3,8,1]的tensor转置为[1,8,3]的tensor
                prediction = prediction.permute(2,1,0)[0]
                for batch_idx, path in enumerate(batch_path):
                    sub_metadata_list.append({
                        path:prediction[batch_idx].cpu().tolist()
                        })
        except Exception as e:
            print("An error occurred:", str(e))
            continue

    save_metadata_path = "aesthetic_score"
    sub_path = f"{save_metadata_path}/score_{args.local_rank}.json"
    with open(sub_path, 'w') as f:
        json.dump(sub_metadata_list, f,indent=4)

    dist.barrier()
# TODO 改成一个一个json 保存；
    if args.local_rank == 0:
        all_caption = []
        for i in range(args.world_size):
          with open(f"{save_metadata_path}/score_{i}.json", 'r') as f:
              metadata_list = json.load(f)
              all_caption.extend(metadata_list)
        with open(f"{save_metadata_path}/all_score.json", 'w') as f:
          json.dump(all_caption, f)
        print(f"processing time:{time.time()-start_time}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
    parser.add_argument('--video_path', default='/aifs4su/mmdata/rawdata/videogen/macvid/', help='Path to the video folder')
    parser.add_argument('--metadata_path', default='video_dataset_85.json', help='metadata file name. Please keep in form of video_dataset_85.json')
    parser.add_argument('--num_frames', default=3, help='number of frames extract from one clip video')
    parser.add_argument('--batch_size', default=8, type=int, help='inference batch size')
    parser.add_argument('--local-rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--world_size', default=6, type=int, help='Number of GPUs for distributed training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of cpu workers for dataloader')
    parser.add_argument('--gpu_ids', default='0', help='devices')

    args = parser.parse_args()

    main(args)

