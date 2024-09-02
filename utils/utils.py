import json
import cv2
import csv
from PIL import Image
import importlib


def get_video_resolution(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Video open error, not found or file broken")
    _, frame = video.read()
    height, width, _ = frame.shape
    video.release()
    return width, height

def get_frames_rate(video_path,sample_rate):
    """
    TODO adding more get_frames_policy
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration= total_frames/fps
    total_frames=total_frames//sample_rate
    frames = []
    
    for i in range(total_frames):
        index=i*sample_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames,duration

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def getImageFromVideo(self, clip_path, num_frames):
    """
    TODO adjust the geting image policy
    """
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
        raise Exception(f"Failed to open video file {clip_path}.")
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def collect_metadata(remove_captions_folder=False):
    # TODO captions/video_dataset_0 --> captions/all/video_dataset_0.json
    # 
    pass

def sort_metadata():
    # TODO captions/all/video_dataset_0.json --> captions/video_dataset_0
    pass

# def load_info(metadata, filename, key):
#     """"
#     find metadata/video_dataset_x/metasadg.json
#     if no metasadg.json: 跑完存进去
#     if key in metasadg.json: 跑过了，跳过
#     else: 跑完之后加进去再存进去
#     """