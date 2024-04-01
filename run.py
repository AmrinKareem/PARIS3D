import os
import torch
import json
import time
import sys
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.seg_lisa import mask2seg
from src.paris3d_inference import paris3d_inference, create_model

def Infer(category, part_names,  model, clip_image_processor, transform, tokenizer, xyz, pc_idx, screen_coords, superpoint, save_dir, sp_dir):

    print("[PARIS3D inference...]")
    predictions = paris3d_inference(model, clip_image_processor, transform, tokenizer, save_dir, part_names, category, sp_dir, precision="fp16", num_views=10, use_mm_start_end=True,
                    save_pred_img=True, save_individual_img=False, save_pred_json=False)

    print('[converting bbox to 3D segmentation...]')
    sem_seg = mask2seg(xyz, superpoint, predictions, screen_coords, pc_idx, part_names, save_dir)
    print("[finish!]")
    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json"))    
    categories = partnete_meta.keys()
    print("[loading PARIS3D model...]")
    model, clip_image_processor, transform, tokenizer = create_model(version="Amrinkar/PARIS3D", model_max_length=512, precision="fp16", load_in_4bit=False, load_in_8bit=False, local_rank=0, image_size=1024, legacy=False)
    for category in categories:

        models = os.listdir(f"test/{category}")
        part_names = partnete_meta[category]
        for modell in models:
            io = IO()
            sp_dir=f'test/{category}/{modell}'   
            input_pc_file = f"test/{category}/{modell}/pc.ply"
            print(f"[normalizing input point cloud...] of {input_pc_file}")
            save_dir=f'test/{category}/{modell}'
            print("[creating tmp dir...]")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
            else:
                device = torch.device("cpu")
            os.makedirs(save_dir, exist_ok=True)
            xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
            print("[rendering input point cloud...]")
            # img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device)
            pc_idx = np.load(f"{sp_dir}/idx.npy", allow_pickle=True)
            screen_coords = np.load(f"{sp_dir}/coor.npy", allow_pickle=True)
            superpoint = np.load(f"{sp_dir}/sp.npy", allow_pickle=True)
            Infer(category, part_names, model, clip_image_processor, transform, tokenizer, xyz, pc_idx, screen_coords, superpoint, save_dir, sp_dir)