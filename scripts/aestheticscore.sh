
# Get the pretrained weight
# mkdir ./models/improved-aesthetic-predictor
# wget  -P ./models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth
weight_path=$(pwd)/models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth
export PYTHONPATH=$PYTHONPATH:../
world_size=1


python -m torch.distributed.launch --nproc_per_node=$world_size --master_port=28500 \
evaluations/aesthetic_score.py \
--world_size $world_size \
--num_workers 1 \
--gpu_ids '0' \
--batch_size 2 \
--video_path '/aifs4su/mmdata/rawdata/videogen/macvid/videos' \
--metadata_path '/aifs4su/mmdata/rawdata/videogen/macvid/metadata/all/video_dataset_85.json' \
--weight_path $weight_path
