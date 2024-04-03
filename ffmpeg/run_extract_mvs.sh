#!/bin/bash
work_dir="`pwd`"

folder=$1
out_folder=$2
mkdir -p $out_folder
start_time=$(date +%s)

for file in "$folder"/*.mp4; do
    if [[ -f "$file" ]]; then
        filename=$(basename -- "$file")
        filename="${filename%.mp4}"
        echo "$(./extract_mvs $file | head -n 1) $filename" >> ${out_folder}/mvs_scores.txt
    fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total mvs processing time: $duration s"