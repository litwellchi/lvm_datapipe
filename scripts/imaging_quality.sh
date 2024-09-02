# Get the pretrained weight
# mkdir ./models/pyiqa_model
# wget  -P ./models/pyiqa_model https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
model_path=$(pwd)/models/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth

python imaging_quality.py \
--vid_dir  \
--metadata_path  \
--model_path ./models/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth \
--out_dir  \
--mp_no 0 \
--num_process 1 \
--mode shorter 