# lvm_datapipe
data pipeline code of large video generation model
## Metadata formats
```
{
"basic": {
    video_id: "string",  # clip属于哪个video 
    video_path: string, # source video path 
    video_duration: float(s),
    video_resolution: [height, weight] 
    video_fps: int,
    clip_id: "string",
    clip_path: string, 
    clip_duration: float(s),
    clip_start_end_idx: [int, int], # clip在source video中的起始帧和结束帧的index (count from 0)
    optimal_score:float
    }
"scene":{
    captions: "Describe the content of the video clip."
    place: "some keyword descriptions"
    background: "some keyword descriptions"
    style: "some keyword descriptions"
    num_of_objects: int
    objects: [
      {
        category: "",         # noun: human,dog,ect...
        action: "",              # verb: run, dance, play guita, ...
        action_speed: "",  # very slow/slow/medium/fast/very fast
        },
      ] # list length is equal to the num_of_objects.
    }
"camera":{
    view_scale: long shot/full shot/medium shot/close-up shot/extreme close-up shot
    movement: static shot, pans and tilts shot, zoom in/zoom out/zoom in and zoom out
    speed: very slow/slow/medium/fast/very fast
    }
 "misc":{}
}
```
## file structure
Please structure the dataset as follows:
```
|video_dataset_0
    |clip1.mp4
    |clip2.mp4
    |...
    |label.json
 |video_dataset_1
    |clip1.mp4
    |clip2.mp4
    |...
    |label.json
```
For each `video_dataset_x` folder, it should contain at most 1 million clips, and less than 1Tb file size after compress.
