import easyocr
import time
import numpy
import glob
import cv2
import os
import json
from PIL import Image,ImageDraw
import argparse


def save_checkpoint(n,no):
    checkpoint = {
        'n': n
    }
    with open(f'checkpoint{no}.json', 'w') as file:
        json.dump(checkpoint, file)

def load_checkpoint(no):
    if os.path.exists(f'checkpoint{no}.json'):
        with open(f'checkpoint{no}.json', 'r') as file:
            checkpoint = json.load(file)
        return checkpoint['n']
    else:
        return 0

def get_frames(video_path,sample_rate):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration= total_frames/fps
    total_frames=total_frames//sample_rate
    frames = []
    
    for i in range(total_frames):
        index=i*sample_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    return frames,duration


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

    return (area/1280/720/len(frames))
    
def process(num,no,sample_rate,video_paths):
    output_file=f'output_10frame{no}.json'
    n=load_checkpoint(no)

    with open(output_file, "a") as file:
        if os.path.getsize(output_file) == 0:
            file.write("[")
        else:
            file.seek(file.tell() - 1) 

        if no!=num-1:
            paths=video_paths[len(video_paths)//num*no+load_checkpoint(no):len(video_paths)//num*(no+1)]
            length=len(video_paths)//num
        else:
            paths=video_paths[len(video_paths)//num*no+load_checkpoint(no):]
            length=len(video_paths)-len(video_paths)//num*no

        for video in paths:
            time_start=time.time()
            frames,duration=get_frames(video,sample_rate)
            score=calculate(frames)

            result = {
                        'id': os.path.splitext(os.path.basename(video))[0],
                        'OCR_Score': score,
                        'duration': duration,
                        'time_taken': time.time()-time_start,
                    }

            json.dump(result, file,indent=4)
            if n<length - 1:
                file.write(",\n")
            print(n)
            n+=1
            save_checkpoint(n,no)
        file.write("]")



if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', default=1, type=int)
    parser.add_argument('--mp_no', default=0, type=int)
    parser.add_argument('--sample_rate', default=10, type=int)
    parser.add_argument('--vid_dir', deafault="/project/llmsvgen/share/data/macvid_4s/video_dataset_85", type=str)
    parser.add_argument('--out_dir', type=str)
    # parser.add_argument('--num_process', type=str)
    args = parser.parse_args()
    num=1
    no=0
    sample_rate=10

    reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
    folder_path = "/project/llmsvgen/share/data/macvid_4s/video_dataset_85"  
    video_paths=[]

    video_paths.extend(sorted(glob.glob(f"{folder_path}/*")))
    process(num,no,sample_rate,video_paths)