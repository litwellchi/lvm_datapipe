data pipeline code of large video generation model
## Notification
代码都还在初步阶段，请把运行脚本和代码push上来。后续再继续维护模块化。
```
scene cut:@yatian
Textcaption:@aosong
```
## Metadata formats
For each clip, it should have one json format metadata.
```
{
"basic": {
    video_id: "string",  # clip属于哪个video 
    video_path: string, # source video path 
    video_duration: float(s),
    video_resolution: [height, weight] 
    video_fps: float,
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
    |metadata.json
 |video_dataset_1
    |clip1.mp4
    |clip2.mp4
    |...
    |metadata.json
```
For each `video_dataset_x` folder, it should contain at most 1 million clips, and less than 1Tb file size after compression.

## Storage Position
盐城算力集群公网ip：
```
V100s:36.133.54.47 -p 65022
gpu:/data/shared_zipdata/
cpu:36.138.58.171 -p 65022
cpu data root：/data/zip_cloud/

```
内网拷贝
```
从cpu copy到v100走内网ip大概200兆s： v100s003: 192.168.0.224，cpu001: 192.168.0.190
cpu to v100: scp -P 65022 /data/shared_zipdata/video_dataset_x user@192.168.0.224:/data/zip_cloud/
```
