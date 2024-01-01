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
        trans_frames = [self.transform(frame).unsqueeze(0) for frame in frames]
        batch_frame = torch.cat(trans_frames, dim=0)
        return batch_frame, idx

    def getImageFromVideo(self, clip_path, num_frames):
        try:
            cap = cv2.VideoCapture(clip_path)
            frame_list = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in [0, frame_count // 2, frame_count - 1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = cap.read()
                frame_list.append(Image.fromarray(frame).convert("RGB"))

            return frame_list

        except:
            raise Exception("Failed to open video file.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
    parser.add_argument('--video_path', default='freeguy_test', help='Path to the video folder')
    parser.add_argument('--num_frames', default=3, help='number of frames extract from one clip video')
    parser.add_argument('--batch_size', default=8, help='inference batch size')
    parser.add_argument('--coca_path', default='coca/open_clip_pytorch_model.bin', help='Path to the coca weight file')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='devices')
    args = parser.parse_args()

    metadata_path = os.path.join(args.video_path, 'metadata.json')
    save_metadata_path = metadata_path.replace('metadata', 'metadata_catpion')
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    open_clip_transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained=args.coca_path,
        device="cuda"
    )
    
    # TODO seems have some bug
    # Use DataParallel for multi-GPU support
    device_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).module

    dataset = VideoDataset(metadata_list, args.video_path, args.num_frames, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    start_time = time.time()

    for batch_frame, idx in tqdm.tqdm(dataloader):
        batch_frame = batch_frame.view(-1, *batch_frame.shape[2:]).to("cuda")        
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = model.generate(batch_frame)
        generated = generated.view(-1, args.num_frames, generated.shape[-1])
        
        for count in range(idx.shape[0]): 
          result_list = [open_clip.decode(generated[count][i]).split("<end_of_text>")[0].replace("<start_of_text>", "") for i in
                        range(args.num_frames)]
          metadata_list[idx[count]]['misc']['frame_caption'] = result_list

    with open(save_metadata_path, 'w') as f:
        json.dump(metadata_list, f)

    end_time = time.time()
    print(end_time - start_time)
