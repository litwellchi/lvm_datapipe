import easyocr
import time
import numpy
import glob
import cv2
import os
import json
import argparse
import tqdm
from PIL import Image,ImageDraw

def get_frames(video_path,sample_rate):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames=total_frames//sample_rate
    frames = []
    
    for i in range(total_frames):
        index=i*sample_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    return frames

def calculate(frames):
    area=0
    for img in frames:
        a=time.time()
        horizontal_list,free_list = reader.detect(img)
        horizontal_list=horizontal_list[0]
        free_list=free_list[0]
        
        for box in horizontal_list:
            if box!=[]:
                x=box[1]-box[0]
                y=box[3]-box[2]
                area+=x*y
        for box in free_list:
            if box!=[]:
                x = [point[0] for point in box]
                y = [point[1] for point in box]
                area+= 0.5 * abs((x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]) - (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0]))  
    
    w, h = frames[0].shape[1], frames[0].shape[0]
    return (area/w/h/len(frames))
    
def process(args):
    num=args.num_process
    no=args.mp_no
    sample_rate=args.sample_rate

    with open(args.metadata_path, 'r') as f:
        metadata_list = json.load(f)
    length=len(metadata_list)
    if no!=num-1:
        metadata_list=metadata_list[length//num*no:length//num*(no+1)]
    else:
        metadata_list=metadata_list[length//num*no:]
    t=time.time()
    clips=[data for data in metadata_list if not os.path.exists(args.out_dir+'/'+data['basic']['clip_id']+'.json') or os.path.getsize(args.out_dir+'/'+data['basic']['clip_id']+'.json') == 0]   
    
    for clip in tqdm.tqdm(clips):
        try:
            clip_path=args.vid_dir+'/'+clip['basic']['clip_path']
            frames=get_frames(clip_path,sample_rate)
            if frames!=[]:
                score=calculate(frames)
                clip['scene']['ocr_score']=score
                with open(args.out_dir+'/'+clip['basic']['clip_id']+'.json','w') as file:
                    json.dump(clip, file, indent=4)
        except Exception as e:
            print("An error occurred:", str(e))
            print("metadata:",args.metadata_path," mp_no:",args.mp_no,"stopped")
            exit()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", type=str, default="")
    parser.add_argument("--num_process", type=int, default=2)
    parser.add_argument("--mp_no", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=10)
    parser.add_argument("--out_dir",type=str,default=".")
    parser.add_argument("--metadata_path",type=str,default="")
    # parser.add_argument("--ckpt_path",type=str,default="./checkpoints")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
    
    process(args)
