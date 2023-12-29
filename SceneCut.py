import os
import subprocess
import json
import tqdm
import multiprocessing as mp

import argparse
from tqdm import tqdm
# Standard PySceneDetect imports:
from scenedetect import open_video
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg


def get_video_resolution(video_path):
    command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'json', video_path]
    result = subprocess.check_output(command)
    res_data = json.loads(result)
    width = res_data['streams'][0]['width']
    height = res_data['streams'][0]['height']
    return [int(height), int(width)]

def find_scenes(video_path, threshold=27.0):
    # Create our video & scene managers, then add the detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
            ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def main(vid_dir, out_dir, file_list, queue):
    threshold = 30.0

    metas = []
    for vid_file in tqdm(file_list):
        if '.' in vid_file:
            vid_name, vid_ext = vid_file.rsplit('.', 1)
        else:
            continue
        if vid_ext in ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg']:
            vid_path = os.path.join(vid_dir, vid_file)
            scenes = find_scenes(vid_path, threshold=threshold)
            for index, scene in enumerate(scenes):
                metadata = {'basic': {}, 'scene': {}, 'camera': {}, 'misc': {}}
                metadata['basic']['video_id'] = vid_name
                metadata['basic']['video_path'] = os.path.join(os.path.basename(vid_dir), vid_file)
                metadata['basic']['video_resolution'] = get_video_resolution(vid_path)
                metadata['basic']['video_duration'] = scenes[-1][1].get_seconds()
                metadata['basic']['video_fps'] = scenes[0][0].get_framerate()
                metadata['basic']['clip_id'] = f'{vid_name}_{"%07d" % index}'
                metadata['basic']['clip_path'] = f"{os.path.basename(out_dir)}/{metadata['basic']['clip_id']}.mp4"
                metadata['basic']['clip_duration'] = (scene[1] - scene[0]).get_seconds()
                metadata['basic']['clip_start_end_idx'] = [scene[0].get_frames(), scene[1].get_frames()]
                metadata['basic']['optimal_score'] = None
                split_video_ffmpeg(vid_path, [scene], f"{out_dir}/{metadata['basic']['clip_id']}.mp4")

                metas.append(metadata)

    queue.put(metas)

def run__process(vid_dir, out_dir, num_process):
    process = []
    results = []
    queue = mp.Queue()

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isabs(vid_dir):
        vid_dir = os.path.abspath(vid_dir)

    file_list = os.listdir(vid_dir)
    chunk_size = len(file_list) // num_process if len(file_list) >= num_process else len(file_list)
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    for chunk in chunks:
        process.append(mp.Process(target=main, args=(vid_dir, out_dir, chunk, queue)))
    [p.start() for p in process]  # 开启了len(threshold_list)个进程
    [p.join() for p in process]
    for _ in process:
        results.extend(queue.get())

    out_json_path = os.path.join(out_dir, 'metadata.json')
    with open(out_json_path, 'w+') as oj:
        json.dump(results, oj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--num_process', type=str)
    args = parser.parse_args()

    run__process(args.vid_dir, args.out_dir, int(args.num_process))




