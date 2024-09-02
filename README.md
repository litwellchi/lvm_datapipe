
# data pipeline code of MMTrail-20M

## 🎞MMTrail: A Multimodal Trailer Video Dataset with Language and Music Descriptions

MMTrail is a large-scale multi-modality video-language dataset with over 20M trailer clips, featuring high-quality multimodal captions that integrate context, visual frames, and background music, aiming to enhance cross-modality studies and fine-grained multimodal-language model training.
**In short, we provided 2M+ LLaVA Video captions, 2M+ Music captions, and 60M+ Coca frame captions for 27.1khrs of Trailer videos.**


[![Arxiv](https://img.shields.io/badge/Arxiv-2407.20962-red)](https://arxiv.org/abs/2407.20962)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://mattie-e.github.io/MMTrail/)
[![🤗Huggingface](https://img.shields.io/badge/🤗-Huggingface-yellow)](https://huggingface.co/datasets/litwell/MMTrail-20M)

<div align='center'>
<img src="examples/teaser.png" class="interpolation-image" alt="teaser." height="96%" width="96%" />
</div>

## Notification
Example code here.

## Update
- [ ] llava captioning and tags generation
- [ ] code format check 
- [x] data loader consistency 
- [x] OCR 
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
[
  {
      'video_id': 'zW1-6V_cN8I',                 # Video ID in MMTrail
      'video_path': 'group_32/zW1-6V_cN8I.mp4',                       # Relative path of the dataset root path
      'video_duration': 1645.52,               # Duration of the video
      'video_resolution': [720, 1280],
      'video_fps': 25.0, 
      'clip_id': 'zW1-6V_cN8I_0000141',           # Clip ID
      'clip_path': 'video_dataset_32/zW1-6V_cN8I_0000141.mp4',          # Relative path of the dataset root path
      'clip_duration': 9.92,            # Duration of the clip itself
      'clip_start_end_idx': [27102, 27350],     # Start frame_id and end frame_id
      'image_quality': 45.510545094807945,      # Image quality score
      'of_score': 6.993135,       # Optical flow score
      'aesthetic_score': [4.515582084655762, 4.1147027015686035, 3.796849250793457], 
      'music_caption_wo_vocal': [{'text': 'This song features a drum machine playing a simple beat. A siren sound is played on the low register. Then, a synth plays a descending lick and the other voice starts rapping. This is followed by a descending run. The mid range of the instruments cannot be heard. This song can be played in a meditation center.', 'time': '0:00-10:00'}],  # Music description of the background music without vocal (human voice).
      'vocal_caption': 'I was just wondering...' # Speech recongitation.
      'frame_caption': ['two people are standing in a room under an umbrella . ', 'a woman in a purple robe standing in front of a man . ', 'a man and a woman dressed in satin robes . '],  # Coca caption of three key frame
      'music_caption': [{'text': 'This music is instrumental. The tempo is medium with a synthesiser arrangement and digital drumming with a lot of vibrato and static. The music is loud, emphatic, youthful, groovy, energetic and pulsating. This music is a Electro Trap.', 'time': '0:00-10:00'}] # Music description of the background music.
      'objects': [' bed', 'Woman', ' wall', ' pink robe', ' pillow'], 
      'background': 'Bedroom', 
      'ocr_score': 0.0, 
      'caption': 'The video shows a woman in a pink robe standing in a room with a bed and a table, captured in a series of keyframes that show her in various poses and expressions.',  # Caption generation from LLaVA and rewrite by LLAMA-13B
      'polish_caption': 'A woman in a pink robe poses and expresses herself in various ways in a room with a bed and a table, capturing her graceful movements and emotive facial expressions.',  # Polished caption generation from LLaVA and rewrite by LLAMA-13B
      'merge_caption': 'In a cozy bedroom setting, a stunning woman adorned in a pink robe gracefully poses and expresses herself, her movements and facial expressions captured in a series of intimate moments. The scene is set against the backdrop of a comfortable bed and a table, with an umbrella standing in a corner of the room. The video features two people standing together under the umbrella, a woman in a purple robe standing confidently in front of a man, and a man and woman dressed in satin robes, all set to an energetic and pulsating electro trap beat with a synthesiser arrangement and digital drumming. The music is loud and emphatic, capturing the youthful and groovy vibe of the video.'# The final description of the video. It is the merge of all above captions, and merged by LLaMA
    }
  }
]
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
python SceneCut.py --vid_dir {PATH_TO_RAW_VIDEOS} --out_dir {PATH_TO_VIDEO_CLIPS} --num_process {PROCESS_NUM} 
# choose a process number best for your device
```
### coca.py
```
python -m torch.distributed.launch --nproc_per_node={PROCESS_NUM_PER_NODE} lvm_datapipe/coca.py --video_path {PATH_TO_VIDEO_CLIPS}  --world_size {WORLD_SIZE} --batch_size {BATCH_NUM} --num_workers {WORKER_NUM}
```
### aesthetic_score.py
```
python -m torch.distributed.launch --nproc_per_node={PROCESS_NUM_PER_NODE} aesthetic_score.py --world_size {WORLD_SIZE}
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
python -m torch.distributed.launch --nproc_per_node={PROCESS_NUM_PER_NODE} llava_caption.py
```
