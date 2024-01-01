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


class MACVDataset(Dataset):
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

