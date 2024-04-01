import json
import numpy as np
import os
import open3d as o3d

def calc_iou(pred, gt) -> float:
    I = np.logical_and(pred, gt).sum()
    U = np.logical_or(pred, gt).sum()
    iou = I / U * 100
    return iou

partnete_meta = json.load(open("PartNetE_meta.json"))
categories = partnete_meta.keys()
tot_miou = 0

for category in categories:
    models = os.listdir(category) # list of models
    part_names = partnete_meta[category]
    cnt = np.zeros(len(part_names))
    cnt_iou = np.zeros(len(part_names))

    for model in models:
    # load gt label
        gt_sem_label = np.load(f"{category}/{model}/label.npy", allow_pickle=True).item()['semantic_seg']
        for i, part in enumerate(part_names):
            if (gt_sem_label==i).sum() == 0:
                continue
            # load predictions
            pcd = o3d.io.read_point_cloud(f"{category}/{model}/semantic_seg/{part}.ply")
            #calculates a binary mask by summing the colors along the last axis and checking if they are greater than zero
            sem_pred = np.asarray(pcd.colors).sum(-1) > 0 
            iou = calc_iou(sem_pred, gt_sem_label==i)
            cnt[i] += 1
            cnt_iou[i] += iou

        part_miou = cnt_iou / cnt
        print(f"{category}", list(zip(part_names, part_miou)))
        tot_miou += part_miou.mean()

print(f"mIoU: {tot_miou / len(categories)}")
