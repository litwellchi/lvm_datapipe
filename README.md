data pipeline code of large video generation model
## Notification
- [ ] code format check 
- [ ] data loader consistency 
- [ ] OCR 
- [x] Code review
- [x] optical flow
- [x] aesthetics score
- [x] llava caption
- [x] llama2 caption summary
- [x] scence cut
- [x] metadata format

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

## Running
### Environment setup
```
pip install -r requirements.txt
```

### SceneCut.py
```
python SceneCut.py --vid_dir /data/shared_zipdata/group_{} --out_dir /data/shared_zipdata/video_dataset_{}/ --num_process 60
```
### coca.py
```
python -m torch.distributed.launch --nproc_per_node=8 lvm_datapipe/coca.py --video_path /home/xiaowei/lvm_datapipe/group_1_mini_clips  --world_size 8 --batch_size 20 --num_workers 4
```
### Running OFScore_with_v2d.py
previous steps: `conda activate vid`
 1. OF:
```
work_dir: lvm_datapipe/
python OFScore_with_v2d.py --input_folder {vid_dir} --output_folder {vid_dir}_of
out_path: {vid_dir}_of/OFresult.json
```
2. MVS:
```
work_dir: ffmpeg-6.1.1/
bash run_extract_mvs.sh 
out_path: {vid_dir}_of/mvs_scores.txt
```
### result analysis & visualization of Pie Graph
```
python analyze_vids.py 
```
