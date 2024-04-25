# Get the pretrained weight
# mkdir ./models/pyiqa_model
# wget  -P ./models/pyiqa_model https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
model_path=$(pwd)/models/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth

python imaging_quality.py \
--vid_dir /project/llmsvgen/share/data/macvid_4s/videos \
--metadata_path /project/llmsvgen/share/data/macvid_4s/metadata/all/macvid_4s_cp_0.json \
--model_path ./models/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth \
--out_dir /project/llmsvgen/share/data/macvid_4s/metadata/imaging_quality/macvid_4s_cp_0 \
--mp_no 0 \
--num_process 1 \
--mode shorter 