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
## Developer notification
```
TODOs:
data_schema 里面放各种数据格式相关的文件，包括dataloader登
evaluations 里放method的脚本（定义好各种arxiv，必须要有的输入统一为all/metadata.json路径或者caption/*.json，输出路径all/metadata.json）；
models 里为各个方法潜在的git clone的模型，请都把路径写到这个底下
scripts 包括了怎么跑起来单个脚本的完整sh，从环境配置，到run
utils里是各种读写等基本操作；在utils里加入检查metadata和已经跑过的断电重启的function
main.py将会集成成一个统一调度的文件
望周知
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
Dataset format
-macvid
    --video
        -- video_dataset_0
        -- video_dataset_1
        -- video_dataset_x
    --metadata
        -- all
            -- video_dataset_0.json 
            -- video_dataset_1.json 
            -- video_dataset_2.json 
        -- video_dataset_0 #one json for one clip
            -- clipidxaasd.json
            -- clipidasd2e.json
        -- video_dataset_1
            -- clipidxaasd.json
            -- clipidasd2e.json
        -- video_dataset_x  
            -- clipidxaasd.json
            -- clipidasd2e.json
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
### aesthetic_score.py
```
python -m torch.distributed.launch --nproc_per_node=8 aesthetic_score.py --world_size 8
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

# llava
```
git clone https://github.com/haotian-liu/LLaVA.git
mv LLaVA llava
python -m torch.distributed.launch --nproc_per_node=7 llava_caption.py
```