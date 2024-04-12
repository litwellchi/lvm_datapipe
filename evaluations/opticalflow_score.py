"""
TODO: data format consistance

"""

import os
import argparse
import json

# previous steps:
##  conda activate vid
##  1. OF:
##      work_dir: lvm_datapipe/
##      python OFScore_with_v2d.py --input_folder {vid_dir} --output_folder {vid_dir}_of
##      out_path: {vid_dir}_of/OFresult.json
##  2. MVS:
##      work_dir: ffmpeg-6.1.1/
##      bash run_extract_mvs.sh 
##      out_path: {vid_dir}_of/mvs_scores.txt

def parse_args():
    parser = argparse.ArgumentParser(description='data filtering based on optical-flow scores')
    parser.add_argument('--in_dir ', type=str, help='${vid_dir}_of')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # read scores
    with open(os.path.join(args.in_dir, 'OFresult.json'), 'r') as of_f, open(os.path.join(args.in_dir, 'mvs_scores.txt'), 'r') as mvs_f:
        of = json.load(of_f)
        mvs = mvs_f.read().splitlines()
    
    mvs_scores, of_scores = {}, {} # {clip_name: [score, index]}
    for _ in of:
        if _['meanOF']:
            of_scores[_['key']] = _['meanOF']
    for _ in mvs:
        s, n = _.split(' ')
        if not float(s) >= 0:
            mvs_scores[n] = float(s)
    
    # index, {clip_name: [score, index]}
    of_scores = {k: [v] for k, v in sorted(of_scores.items(), key=lambda item: item[1])}
    of_scores = {k: [v, i] for i, (k, v) in enumerate(of_scores.items())}
    mvs_scores = {k: [v] for k, v in sorted(mvs_scores.items(), key=lambda item: item[1])}
    mvs_scores = {k: [v, i] for i, (k, v) in enumerate(mvs_scores.items())}
    
    # TODO: restore indexes <9424> [above 58 lines]
    # TODO: restore all the normed scores. Normalization: (B2-MIN())/(MAX()-MIN())*20+1 [above 58 lines]
    ok_set, tocut_set = set(), set()
    of_set = of_scores.keys()
    mvs_set = mvs_scores.keys()
    mvs_only = mvs_set - of_set # if is invalid mvs: drop
    all_keys = of_set.union(mvs_only)
    # Filtering 
    ## Diff
    for key in all_keys:
        if not mvs_scores[key][0] < 0.1: 
            ok_set.append(key)
        if key in of_set:
            score_diff = mvs_scores[key][0] - of_scores[key][0]
            index_diff = mvs_scores[key][1] - of_scores[key][1]
            if (index_diff > 5000 and score_diff > 15) or (index_diff < -3000 and score_diff < 1) or \
                (of_scores[key][0] < 0.02) or (0.02 <= of_scores[key][0] < 0.045 and index_diff < 0) or \
                    (mvs_scores[key][0] < 0.45 and index_diff < -160) or (0.45 <= mvs_scores[key][0] < 0.85 and index_diff < -1000):
                continue
            elif -2000 <= index_diff < -3000 and 0 < score_diff <= 1:
                tocut_set.append(key)
            else:
                ok_set.append(key)   



if __name__ == '__main__':
    main()