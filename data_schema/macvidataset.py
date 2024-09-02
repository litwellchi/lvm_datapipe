import os
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset
import yaml

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
        clip_path = os.path.join(self.video_path, f'{metadata["basic"]["clip_path"]}')
        frames = self.getImageFromVideo(clip_path, points=[0.2,0.5,0.8])
        if frames == None: return None, idx
        batch_frame = []
        for frame in frames:
            image = self.transform(frame).unsqueeze(0)
            batch_frame.append(image)
        return batch_frame, idx, metadata["basic"]["clip_path"]
    
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
