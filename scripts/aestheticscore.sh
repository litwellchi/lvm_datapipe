
# Get the pretrained weight
mkdir ./models/improved-aesthetic-predictor
wget  -P ./models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth
weight_path=$(pwd)/models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth
export PYTHONPATH=$PYTHONPATH:../

python -m torch.distributed.launch --nproc_per_node=1 evaluations/aesthetic_score.py \
--world_size 1 \
--weight_path $weight_path