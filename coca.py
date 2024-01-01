import os
import time
import json
import cv2
import torch
import argparse
import tqdm
from PIL import Image
import open_clip
import torchvision.transforms as transforms


def getImageFromVideo(clip_path, nume_frames = 3):
  try:
    cap = cv2.VideoCapture(clip_path)
    frame_list = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # TODO
    for i in [0,frame_count // 2, frame_count - 1]:
      cap.set(cv2.CAP_PROP_POS_FRAMES, i)
      _, frame = cap.read()
      frame_list.append(Image.fromarray(frame).convert("RGB"))
    
    return frame_list

  except:
    Exception("Failed to open video file.")

if __name__ =="__main__":

  parser = argparse.ArgumentParser(description='Extract misc strings from JSON file.')
  parser.add_argument('--video_path', default='freeguy_test', help='Path to the JSON file')
  parser.add_argument('--nume_frames', default=3, help='Path to the JSON file')
  parser.add_argument('--coca_path', default='coca/open_clip_pytorch_model.bin', help='Path to the JSON file')
  parser.add_argument('--gpus', default='0', help='devices')
  args = parser.parse_args()

  metadata_path = os.path.join(args.video_path,'metadata.json')
  save_metadata_path = metadata_path.replace('metadata','metadata_catpion')
  with open(metadata_path,'r') as f:
    metadata_list = json.load(f)

  # open_clip transforms 
  open_clip_transform = transforms.Compose([
      transforms.Resize(size=224),
      transforms.CenterCrop(size=(224, 224)),
      # transforms.ToTensor(),
      transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
  ])

  model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained=args.coca_path,
        device="cuda"
      )
  
  start_time = time.time()
  # TODO build a dataloader
  for metadata in metadata_list:
    clip_path = os.path.join(args.video_path, metadata["basic"]["clip_path"])
    frames = getImageFromVideo(clip_path,args.nume_frames)

    
    trans_frames = [transform(frame).unsqueeze(0).to("cuda") for frame in frames] 

    batch_frame = torch.cat(trans_frames,dim=0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(batch_frame)
          
    result_list = [open_clip.decode(generated[i]).split("<end_of_text>")[0].replace("<start_of_text>", "") for i in range(args.nume_frames)]
    metadata['misc']['frame_caption'] = result_list


  with open(save_metadata_path,'w') as f:
    json.dump(metadata, f)
    
  end_time = time.time()
  print(start_time-end_time)
