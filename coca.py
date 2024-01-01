import os
import time
import json
import cv2
import torch
import argparse
from PIL import Image
import open_clip


def getImageFromVideo(video_path, image_path,num=1):
    if os.path.exists(image_path):
        return
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num==2:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count/2)
    if num==3:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count-1)        
    ret, frame = cap.read()

    cv2.imwrite(f"{image_path}_{num}.jpg", frame)
    cap.release()


parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
parser.add_argument('--json_file', default='./', help='Path to the JSON file')
args = parser.parse_args()

list = os.listdir("/home/aosong/freeguy_test")
caption_list = []
start_time = time.time()
for video in list:
   video_path = os.path.join("/home/aosong/freeguy_test", video)
   image_path = os.path.join("/home/aosong/freeguy_test_image", video.replace(".mp4", ""))
   getImageFromVideo(video_path, image_path,1)
   getImageFromVideo(video_path, image_path,2)
   getImageFromVideo(video_path, image_path,3)
   model, _, transform = open_clip.create_model_and_transforms(
         model_name="coca_ViT-L-14",
         pretrained="/home/aosong/coca/open_clip_pytorch_model.bin",
         device="cuda"
      )

   
   im1 = transform(Image.open(f"{image_path}_1.jpg").convert("RGB")).unsqueeze(0).to("cuda")
   im2 = transform(Image.open(f"{image_path}_2.jpg").convert("RGB")).unsqueeze(0).to("cuda")
   im3 = transform(Image.open(f"{image_path}_3.jpg").convert("RGB")).unsqueeze(0).to("cuda")
   im = torch.cat((im1,im2,im3),dim=0)

   with torch.no_grad(), torch.cuda.amp.autocast():
       generated = model.generate(im)
        
   # caption_list.append(
   #    {"text":open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""),
   #      "video": video})
   print(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
   print(open_clip.decode(generated[1]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
   print(open_clip.decode(generated[2]).split("<end_of_text>")[0].replace("<start_of_text>", ""))

end_time = time.time()
print(start_time-end_time)
