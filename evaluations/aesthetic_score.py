import os
import json
import time
from data_schema.macvidataset import MACVDataset as VideoDataset
from data_schema.macvid import macvid_path_dict
import argparse
import numpy as np
import tqdm
import time
import clip
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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
    
    num=args.process_num
    no=args.mp_no
    length=len(metadata_list)
    if no!=num-1:
        metadata_list=metadata_list[length//num*no:length//num*(no+1)]
    else:
        metadata_list=metadata_list[length//num*no:]
    save_path=macvid_path_dict(args.metadata_path)['metadata_folder']
    metadata_list=[data for data in metadata_list if not os.path.exists(save_path+'/'+data['basic']['clip_id']+'.json')]

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load(args.weight_path)   # load the model you trained previously or the model available in this repo
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
    save_metadata_path =  macvid_path_dict(args.metadata_path)['metadata_folder']
    if not os.path.exists(save_metadata_path):
        os.makedirs(save_metadata_path)
    for batch_frame, idx, save_meta in tqdm.tqdm(dataloader):
        try:
            with torch.no_grad():
                frames = []
                for image in batch_frame:
                    image_features = model2.module.encode_image(image.squeeze(1).to(args.local_rank))
                    im_emb_arr = normalized(image_features.cpu().detach().numpy())
                    frames.append(im_emb_arr)

                frames = torch.tensor(np.array(frames)).to(args.local_rank)
                prediction = model(frames.type(torch.cuda.FloatTensor))

                prediction = prediction.permute(2,1,0)[0]
                # Saving each result
                for batch_idx, path in enumerate(save_meta):
                    sub_metadata_list  = metadata_list[idx[batch_idx]]
                    sub_metadata_list['basic']['optimal_score'] = prediction[batch_idx].cpu().tolist()
                    sub_path = f"{save_metadata_path}/{sub_metadata_list['basic']['clip_id']}.json"
                    with open(sub_path, 'w') as f:
                        json.dump(sub_metadata_list, f,indent=4)
                
        except Exception as e:
            print("An error occurred:", str(e))
            continue
    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
    parser.add_argument('--video_path', default='', help='Path to the video folder')
    parser.add_argument('--metadata_path', default='', help='metadata file name. Please keep in form of video_dataset_85.json')
    parser.add_argument('--num_frames', default=3, help='number of frames extract from one clip video')
    parser.add_argument('--batch_size', default=8, type=int, help='inference batch size')
    parser.add_argument('--local-rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument('--world_size', default=0, type=int, help='Number of GPUs for distributed training')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of cpu workers for dataloader')
    parser.add_argument('--weight_path', default="../models/improved-aesthetic-predictor/ava+logos-l14-linearMSE.pth", type=str, help='')
    parser.add_argument('--gpu_ids', default='0', help='devices')
    parser.add_argument('--process_num',default=1,type=int)
    parser.add_argument('--mp_no',default=0,type=int)
    args = parser.parse_args()

    main(args)
