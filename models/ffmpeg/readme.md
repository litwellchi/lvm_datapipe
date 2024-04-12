# TODO for Optical FLow
1. FFMPEG
1.1 Install ffmpeg. Follow: https://blog.csdn.net/Geek_sun/article/details/113632794.
1.2 Move files under `lvm_datapipe/ffmpeg/` under the real ffmpeg folder. Then compile `extract_mvs.c`.
1.3 Run `run_extract_mvs.sh ${vid_dir} ${vid_dir}_of` under the real ffmpeg folder.
2. Video2dataset
2.1 Make sure under `lvm_datapipe/`
2.2 `conda create -n vid --file requirements.txt`
2.3 `python OFScore_with_v2d.py --input_folder ${vid_dir} --output_folder ${vid_dir}_of`
3. Combining Results and Filtering
3.1 Change corresponding thresholds according to TODO instractions.
3.2 `python optical_flow_filtering.py --in_dir ${vid_dir}_of`