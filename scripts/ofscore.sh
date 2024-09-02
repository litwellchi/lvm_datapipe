#!/bin/bash
root_path="`pwd`"
work_dir=$pwd/models/of_scores
yasm_path=$work_dir/compile/yasm-1.3.0
ffmpeg_path=$work_dir/compile/ffmpeg
mkdir -p models/of_scores
cd models/of_scores
# 下载编译ffmpeg需要的包

wget https://ffmpeg.org//releases/ffmpeg-snapshot.tar.bz2
tar -jxvf ffmpeg-snapshot.tar.bz2

wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar -xvzf yasm-1.3.0.tar.gz

cd yasm-1.3.0/
mkdir -p $yasm_path
./configure --prefix=$yasm_path
make -j64
make install

cd ../ffmpeg
mkdir -p $ffmpeg_path
./configure --yasmexe=$yasm_path/bin/yasm --enable-shared --prefix=$ffmpeg_path
make -j64
make install

# 将默认的extract_mvs.c替换为专用的提取scores的脚本
rm $work_dir/ffmpeg/doc/examples/extract_mvs.c
cp $root_path/models/extract_mvs.c $work_dir/ffmpeg/doc/examples/extract_mvs.c
gcc -o extract_mvs $work_dir/ffmpeg/doc/examples/extract_mvs.c -L$ffmpeg_path/lib -I$ffmpeg_path/include -lavcodec -lavdevice -lavfilter -lavformat -lavutil -lswresample -lm



# 生成光流
cd $root_path
video_path=""
# scores输出的地址
output_path=

# 获取mv_scores，输出一个mvs_scores.txt
sh $root_path/scripts/run_extract_mvs.sh  $work_dir $video_path $output_path
# 获取of_scores,输出到video_path中
python $root_path/evaluations/OFScore_with_v2d.py --input_folder ${video_path} --output_folder $output_path

# 筛选出需要cut掉的视频
# python evaluations/opticalflow_score.py \
# --video_path $video_path \
# --result_path $output_path \
