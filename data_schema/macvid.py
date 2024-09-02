import os
import torch
import json
import random
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
import yaml


metadata = {
    "basic": {
        "video_id": "",                 # Type: str - clip belongs to which video
        "video_path": "",               # Type: str - source video path
        "video_duration": 0.0,          # Type: float (seconds)
        "video_resolution": [],         # Type: list [height, width]
        "video_fps": 0.0,                 # Type: float
        "clip_id": "",                  # Type: str
        "clip_path": "",                # Type: str
        "clip_duration": 0.0,           # Type: float (seconds)
        "clip_start_end_idx": [0, 0],   # Type: list [int, int] - start and end frame indices in the source video (count from 0)
        "optimal_score": 0.0            # Type: float
    },
    "scene": {
        "captions": "",                 # Type: str - describes the content of the video clip
        "place": "",                    # Type: str - keyword descriptions for the place
        "background": "",               # Type: str - keyword descriptions for the background
        "style": "",                    # Type: str - keyword descriptions for the style
        "num_of_objects": 0,            # Type: int
        "objects": [                    # Type: list - length is equal to num_of_objects
            {
                "category": "",         # Type: str - noun: human, dog, etc.
                "action": "",           # Type: str - verb: run, dance, play guitar, etc.
                "action_speed": ""      # Type: str - very slow/slow/medium/fast/very fast
            }
        ]
    },
    "camera": {
        "view_scale": "",               # Type: str - long shot/full shot/medium shot/close-up shot/extreme close-up shot
        "movement": "",                 # Type: str - static shot, pans and tilts shot, zoom in/zoom out/zoom in and zoom out
        "speed": ""                     # Type: str - very slow/slow/medium/fast/very fast
    },
    "misc": {}                          # Type: dict - additional miscellaneous metadata
}

def merged_metadata(metadata_path):
    """
    captions/task_a --> captions/all/task_a.json
    """
    video_path = metadata_path.replace('.json','').replace('metadata/all','videos')
    metadata_folder = video_path.replace("videos","metadata")
    merged_data = []
    for filename in os.listdir(metadata_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(metadata_folder, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                merged_data.append(json_data)
    with open(metadata_path, 'w') as file:
        json.dump(merged_data, file)
    
def sort_metadata(metadata_path):
    """
    captions/all/task_a.json --> captions/task_a
    """
    video_path = metadata_path.replace('.json','').replace('metadata/all','videos')
    metadata_folder = video_path.replace("videos","metadata")
    with open(metadata_path, 'r') as file:
        json_data = json.load(file)
    
    os.makedirs(metadata_folder, exist_ok=True)

    for i, data in enumerate(json_data):
        output_file = os.path.join(metadata_folder, f"{data['basic']['clip_id']}.json")
        with open(output_file, 'w') as file:
            json.dump(data, file)
            
def get_metadata_list(yaml_path):
    # TODO: 断断点重传功能在这里更新
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("DATASET CONFIG:")
    print(config)
    videos = []
    data_root = config['data_root']
    for meta_path in config['META']:
        with open(meta_path, 'r') as f:
            videos = json.load(f)
            for item in videos:
                # TODO conditions
                videos.append(item)
            
    print(f'Number of videos = {len(videos)}')

def macvid_path_dict(metadata_path):
    return{
        'metadata_path':metadata_path,
        'video_path':metadata_path.replace('metadata/all','videos').replace('.json',''),
        'metadata_folder':metadata_path.replace('metadata/all','metadata').replace('.json','')   
    }

    
class MaCVid(Dataset):
    """
    Dataset format: 
    -- macvid_dataset
        -- videos
            -- video_dataset_0
            -- video_dataset_1
            -- video_dataset_x
        -- metadata
            -- all
                -- selected_target.json 
                -- best_ocr_only.json 
                -- others.json 
            -- video_dataset_0 #one json for one clip
                -- clipidxaasd.json
                -- clipidasd2e.json
            -- video_dataset_1
                -- clipidxaasd.json
                -- clipidasd2e.json
            -- video_dataset_x  
                -- clipidxaasd.json
                -- clipidasd2e.json
    """
    def __init__(self,
                 yaml_path,
                 resolution,
                 video_length,
                 frame_stride=4,
                 clip_length=1.0
                 ):
        self.yaml_path = yaml_path
        self.resolution = resolution
        self.video_length = video_length
        self.frame_stride = frame_stride
        self.clip_length = clip_length

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        assert(isinstance(self.resolution, list) and len(self.resolution) == 2)

        self._make_dataset()
    
    def _make_dataset(self):
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        self.videos = []
        self.data_root = self.config['data_root']
        for meta_path in self.config['META']:
            with open(meta_path, 'r') as f:
                videos = json.load(f)
                for item in videos:
                    # TODO conditions
                    self.videos.append(item)
                
        print(f'Number of videos = {len(self.videos)}')

    def __getitem__(self, index):
        while True:
            video_path = os.path.join(self.data_root, self.videos[index]['basic']['clip_path'])
            try:
                video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                return self.__getitem__(index)
    
        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))

        # select random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = list(range(rand_idx, rand_idx+self.video_length))
        frames = video_reader.get_batch(frame_indices)
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        data = {'video': frames, 'caption':self.videos[index]["misc"]['frame_caption'][0]}
        return data
    
    def __len__(self):
        return len(self.videos)