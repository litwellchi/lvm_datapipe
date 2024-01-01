import os
import subprocess
import json
import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed

import argparse
from tqdm import tqdm
# Standard PySceneDetect imports:
from scenedetect import open_video
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

from metadata import MetadataDict


def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def find_breakpoint(metadata_list):
    vid_visited = dict()
    vid_finished = []
    for clip in metadata_list:
        meta_clip = MetadataDict()
        meta_clip.load_from_dict(clip)
        vid_id = os.path.basename(meta_clip.get_value("basic", "video_path"))
        if vid_id not in vid_visited:
            vid_visited[vid_id] = [meta_clip.get_value("basic", "video_duration"), 0.]
        vid_visited[vid_id][1] += meta_clip.get_value("basic", "clip_duration")
        if vid_visited[vid_id][0] - vid_visited[vid_id][1] < 0.5:
            vid_finished.append(vid_id)
    return vid_finished


def main(vid_dir, out_dir, file_list):
    threshold = 30.0

    with open(os.path.join(out_dir, 'metadata.json'), 'a') as out_file:
        for vid_file in tqdm(file_list):
            try:
                if '.' in vid_file:
                    vid_name, vid_ext = vid_file.rsplit('.', 1)
                else:
                    continue
                if vid_ext in ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg']:
                    vid_path = os.path.join(vid_dir, vid_file)
                    scenes = find_scenes(vid_path, threshold=threshold)
                    for index in range(len(scenes)):
                        metadata = MetadataDict()
                        metadata.set_basic_info(index, video_id=vid_name, video_path=vid_path, scenes=scenes,
                                                out_dir=out_dir)
                        split_video_ffmpeg(vid_path, [scenes[index]],
                                           f"{out_dir}/{metadata.get_value('basic', 'clip_id')}.mp4")

                        output_metadata = json.dumps(metadata.to_dict())
                        out_file.write(output_metadata + ', ')
            except Exception as e:
                print("An error occurred:", str(e))
                continue


def run__process(vid_dir, out_dir, num_process):
    if not os.path.isabs(vid_dir):
        vid_dir = os.path.abspath(vid_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, 'metadata.json')
    if os.path.getsize(out_json_path) > 1:
        with open(out_json_path, 'r') as oj:
            try:
                processed_list = json.loads(oj.read())
                finished_list = find_breakpoint(processed_list)
            except:
                finished_list = []
    else:
        finished_list = []


    file_list = os.listdir(vid_dir)
    file_list = list(set(file_list)-set(finished_list))
    if len(file_list) == 0:
        return
    return_flag = 1
    for file in file_list:
        if file.rsplit('.', 1)[-1] in ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg']:
            return_flag = 0
            break
    if return_flag:
        return

    with open(out_json_path, 'w') as oj:
        oj.write('[')

    chunk_size = len(file_list) // num_process if len(file_list) >= num_process else len(file_list)
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    # Use joblib to allocate processes on CPUs
    Parallel(n_jobs=num_process)(delayed(main)(vid_dir, out_dir, chunk) for chunk in chunks)

    with open(out_json_path, 'r') as oj:
        line = oj.readlines()
        out = line[0][:-2] + ']'
    with open(out_json_path, 'w') as oj:
        oj.write(out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--num_process', type=str)
    args = parser.parse_args()

    run__process(args.vid_dir, args.out_dir, int(args.num_process))
