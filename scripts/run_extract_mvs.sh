#!/bin/bash
work_dir=$1

folder=$2
out_folder=$3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$work_dir/compile/ffmpeg/lib
mkdir -p $out_folder
start_time=$(date +%s)

for file in "$folder"/*.mp4; do
    if [[ -f "$file" ]]; then
        filename=$(basename -- "$file")
        filename="${filename%.mp4}"
        echo "$($work_dir/extract_mvs $file | head -n 1) $filename" >> ${out_folder}/mvs_scores.txt
    fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total mvs processing time: $duration s"