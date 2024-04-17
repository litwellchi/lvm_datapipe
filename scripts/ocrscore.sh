#pip install easyocr
#pip install opencv-python
python ocr_score.py \
--vid_dir /project/llmsvgen/share/data/macvid_4s/videos \
--metadata_path /project/llmsvgen/share/data/macvid_4s/metadata/all/macvid_4s_cp_0.json \
--out_dir "/project/llmsvgen/share/data/macvid_4s/metadata/ocrscore" \
--ckpt_path "/project/llmsvgen/share/data/macvid_4s/metadata/ocrscore/checkpoints" \
--sample_rate 10 \
--num_process 1 \
--mp_no 0

