import os
import time

import multiprocessing as mp

import cv2
import argparse
from tqdm import tqdm
# Standard PySceneDetect imports:
from scenedetect import open_video
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg


def find_scenes(video_path, det_type='c', threshold=27.0):
    # Create our video & scene managers, then add the detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    if det_type == 'a':
        scene_manager.add_detector(
            AdaptiveDetector(adaptive_threshold=threshold))
    elif det_type == 'c':
        scene_manager.add_detector(
            ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def main(vid_dir, out_dir, det_type, threshold):
    times = []
    if args.vid_dir[-4:] in ['.mp4', '.mkv', '.mpg', '.mpeg', '.avi', '.rm', '.rmvb', '.mov', '.wmv', '.asf', '.dat']:
        vid_path = os.path.join(args.vid_dir)
        # for threshold in threshold_list:
        t0 = time.time()
        scenes = find_scenes(vid_path, det_type, threshold=threshold)
        t1 = time.time()
        out_dir = os.path.join(args.out_dir, ''.join(os.path.basename(args.vid_dir).split('.')[:-1]) + '_' + str(threshold))
        os.makedirs(out_dir, exist_ok=True)
        for index, scene in enumerate(scenes):
            split_video_ffmpeg(vid_path, [scene], f"{out_dir}/{index + 1:05}.mp4")
        t2 = time.time()
        times.append([threshold, t1-t0, t2-t1])
    else:
        for vid_name in tqdm(os.listdir(args.vid_dir)):
            if vid_name[-4:] not in ['.mp4', '.mkv', '.mpg', '.mpeg', '.avi', '.rm', '.rmvb', '.mov', '.wmv', '.asf', '.dat']:
                continue
            vid_path = os.path.join(args.vid_dir, vid_name)
            # for threshold in threshold_list:
            t0 = time.time()
            scenes = find_scenes(vid_path, det_type, threshold=threshold)
            t1 = time.time()
            out_dir = os.path.join(args.out_dir, vid_name+'_'+str(threshold))
            os.makedirs(out_dir, exist_ok=True)
            for index, scene in enumerate(scenes):
                split_video_ffmpeg(vid_path, [scene], f"{out_dir}/{index + 1:05}.mp4")
            t2 = time.time()
            times.append([threshold, t1 - t0, t2 - t1])

    for tm in times:
        print('Threshold: '+ str(tm[0]) + ' -- FindScenesTime: ' + str(tm[1]) + ', CutScenesTime: ' + str(tm[2]))


def run__process(vid_dir, out_dir, det_type, threshold_list):
    process = []
    for threshold in threshold_list:
        process.append(mp.Process(target=main, args=(vid_dir, out_dir, det_type, threshold)))
    [p.start() for p in process]  # 开启了len(threshold_list)个进程
    [p.join() for p in process]

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--det_type', type=str, help='Including: "c": ContentDetector, "a": AdaptiveDetector')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    threshold_list = []
    if args.det_type == 'a':
        # threshold_list = [5.0]    # best
        threshold_list = [5.0, 8.0, 10.0, 15.0]
    elif args.det_type == 'c':
        # threshold_list = [30.0]   # best
        threshold_list = [30.0, 20.0, 40.0, 50.0, 35.0, ]
                          # 25.0, 45.0, 15.0, 10.0, 27.0, \
                          # 32.0, 37.0, 42.0, 17.0, 22.0, \
                          # 36.0, 33.0, 28.0, 29.0, 31.0, \
                          # 34.0, 38.0, 39.0, 41.0, 60.0]
    run__process(args.vid_dir, args.out_dir, args.det_type, threshold_list)