"""
Dataset format
-macvid
    --video
        -- video_dataset_0
        -- video_dataset_1
        -- video_dataset_x
    --metadata
        -- all
            -- video_dataset_0.json 
            -- video_dataset_1.json 
            -- video_dataset_2.json 
        -- video_dataset_0 #one json for one clip
            -- clipidxaasd.json
            -- clipidasd2e.json
        -- video_dataset_1
            -- clipidxaasd.json
            -- clipidasd2e.json
        -- video_dataset_x  
            -- clipidxaasd.json
            -- clipidasd2e.json
"""


def collect_metadata(remove_captions_folder=False):
    # TODO captions/video_dataset_0 --> captions/all/video_dataset_0.json
    # 
    pass

def sort_metadata():
    # TODO captions/all/video_dataset_0.json --> captions/video_dataset_0
    pass

def load_info(metadata, filename, key):
    """"
    find metadata/video_dataset_x/metasadg.json
    if no metasadg.json: 跑完存进去
    if key in metasadg.json: 跑过了，跳过
    else: 跑完之后加进去再存进去
    """