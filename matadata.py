from typing import List, Tuple, Any, Union
import os
import json
import subprocess

# Standard PySceneDetect imports:
from scenedetect.frame_timecode import FrameTimecode


def get_video_resolution(video_path):
    command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'json', video_path]
    result = subprocess.check_output(command)
    res_data = json.loads(result)
    width = res_data['streams'][0]['width']
    height = res_data['streams'][0]['height']
    return [int(height), int(width)]


class MetadataDict:
    def __init__(self):
        self.metadata = {
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

    def set_basic_info(self, index: int, video_id: str, video_path: str, scenes: List[List[FrameTimecode]],
                       out_dir: str, optimal_score: float = None):
        scene = scenes[index]
        self.metadata["basic"]["video_id"] = video_id
        self.metadata["basic"]["video_path"] = os.path.join(os.path.basename(os.path.dirname(video_path)), os.path.basename(video_path))
        self.metadata["basic"]["video_duration"] = scenes[-1][1].get_seconds()
        self.metadata["basic"]["video_resolution"] = get_video_resolution(video_path)
        self.metadata["basic"]["video_fps"] = scenes[0][0].get_framerate()
        self.metadata["basic"]["clip_id"] = f'{video_id}_{"%07d" % index}'
        self.metadata["basic"]["clip_path"] = f"{os.path.basename(out_dir)}/{self.metadata['basic']['clip_id']}.mp4"        # TODO: no dirname
        self.metadata["basic"]["clip_duration"] = (scene[1] - scene[0]).get_seconds()
        self.metadata["basic"]["clip_start_end_idx"] = [scene[0].get_frames(), scene[1].get_frames()]
        self.metadata["basic"]["optimal_score"] = optimal_score

    def _set_basic_info(self, video_id: str, video_path: str, video_duration: float,
                       video_resolution: List[int], video_fps: int, clip_id: str,
                       clip_path: str, clip_duration: float, clip_start_end_idx: List[int],
                       optimal_score: float):
        self.metadata["basic"]["video_id"] = video_id
        self.metadata["basic"]["video_path"] = video_path
        self.metadata["basic"]["video_duration"] = video_duration
        self.metadata["basic"]["video_resolution"] = video_resolution
        self.metadata["basic"]["video_fps"] = video_fps
        self.metadata["basic"]["clip_id"] = clip_id
        self.metadata["basic"]["clip_path"] = clip_path
        self.metadata["basic"]["clip_duration"] = clip_duration
        self.metadata["basic"]["clip_start_end_idx"] = clip_start_end_idx
        self.metadata["basic"]["optimal_score"] = optimal_score

    def set_scene_info(self, captions: str, place: str, background: str, style: str,
                       num_of_objects: int, objects: List[dict]):
        self.metadata["scene"]["captions"] = captions
        self.metadata["scene"]["place"] = place
        self.metadata["scene"]["background"] = background
        self.metadata["scene"]["style"] = style
        self.metadata["scene"]["num_of_objects"] = num_of_objects
        self.metadata["scene"]["objects"] = objects

    def set_camera_info(self, view_scale: str, movement: str, speed: str):
        self.metadata["camera"]["view_scale"] = view_scale
        self.metadata["camera"]["movement"] = movement
        self.metadata["camera"]["speed"] = speed

    def set_misc_info(self, misc_info: dict):
        self.metadata["misc"] = misc_info

    def load_from_dict(self, metadata_dict: dict):
        if "basic" in metadata_dict:
            self._set_basic_info(**metadata_dict["basic"])
        if "scene" in metadata_dict:
            try:
                self.set_scene_info(**metadata_dict["scene"])
            except:
                self.set_scene_info('', '', '', '', 0, [])
        if "camera" in metadata_dict:
            try:
                self.set_camera_info(**metadata_dict["camera"])
            except:
                self.set_camera_info('', '', '')
        if "misc" in metadata_dict:
            try:
                self.set_misc_info(metadata_dict["misc"])
            except:
                self.set_misc_info({})


    def get_metadata(self):
        return self.metadata

    def get_value(self, section: str, key: str) -> Any:
        if section in self.metadata and key in self.metadata[section]:
            return self.metadata[section][key]
        else:
            return None

    def update_value(self, section: str, key: str, value: Union[str, int, float, List[Any], dict]) -> bool:
        if section in self.metadata and key in self.metadata[section]:
            self.metadata[section][key] = value
            return True
        else:
            return False

    def to_dict(self) -> dict:
        return self.metadata

