import os
import cv2
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_colored_pc, get_iou
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def get_point_iou(pc1, pc2):
    I = np.logical_and(pc1, pc2)
    U = np.logical_or(pc1, pc2)
    if U.sum() < 10:
        return 0
    return I.sum() / U.sum()
def check_pc_within_seg_mask(mask, pc_coord):
    '''
    Args:
        mask: [H, W]. Segmentation mask
        pc_coord: [N, 2]. pc -> screen mapping
    Return:
        flag: [N]. Whether pc is belong to the category of mask
    '''
    flag = mask[pc_coord[:, 1], pc_coord[:, 0]]
    return flag

def check_pc_within_bbox(x1, y1, x2, y2, pc):  
    flag = np.logical_and(pc[:, 0] > x1, pc[:, 0] < x2)
    flag = np.logical_and(flag, pc[:, 1] > y1)
    flag = np.logical_and(flag, pc[:, 1] < y2)

    return flag

def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

def get_union(f, x): # union-find
    if f[x] == x:
        return x
    f[x] = get_union(f, f[x])
    return f[x]

def calc_sp_connectivity(xyz, superpoints, thr=0.02): 
# calculate connectivity (bounding box adjacency) between superpoints
    n = len(superpoints)
    X_min, X_max = [], []
    for i in range(n):
        X_min.append(xyz[superpoints[i], :].min(axis=0))
        X_max.append(xyz[superpoints[i], :].max(axis=0))
    X_min = np.array(X_min)
    X_max = np.array(X_max)
    A = (X_min.reshape(n, 1, 3) - X_max.reshape(1, n, 3)).max(axis=2)
    A = np.maximum(A, A.transpose())
    connectivity = A < thr
    return connectivity

def mask2seg(xyz, superpoint, preds, screen_coor_all, point_idx_all, part_names, save_dir,
            num_view=10, visualize=True):
    print("semantic segmentation...")

    n_category = len(part_names)
    n_sp = len(superpoint)
    sp_visible_cnt = np.zeros(n_sp) #visible points for each superpoint
    sp_bbox_visible_cnt = np.zeros((n_category, n_sp)) 
    #visible points of superpoint j that are covered by at least a bounding box of category i
    preds_per_view = [[] for i in range(num_view)]
    for pred in preds:
        preds_per_view[pred["image_id"]].append(pred)
    in_box_ratio_list = [[[] for j in range(n_sp)] for i in range(n_category)] #used for instance segmentation
    visible_pts_list = []
    for i in range(num_view):
        screen_coor = screen_coor_all[i] #2D projected location of each 3D point
        point_idx = point_idx_all[i] #point index of each 2D pixel
        visible_pts = np.unique(point_idx)[1:] # the first one is -1
        visible_pts_list.append(visible_pts)
        valid_preds = []
        for pred in preds_per_view[i]:
            nonzero_coords = np.argwhere(pred["mask"])
            if len(nonzero_coords) == 0:
                continue
            x1 = nonzero_coords[:, 1].min()
            x2 = nonzero_coords[:, 1].max()
            y1 = nonzero_coords[:, 0].min()
            y2 = nonzero_coords[:, 0].max()
    
            if check_pc_within_bbox(x1, y1, x2, y2, screen_coor).mean() < 0.98: 
            #ignore bbox covering the whole objects
                valid_preds.append(pred)
   
        for k, sp in enumerate(superpoint):
            sp_visible_pts = intersection(sp, visible_pts) 
            sp_visible_cnt[k] += len(sp_visible_pts) # Eg: array([2., 0., 0., ..., 0., 0., 0.]). 2nd iteration: array([  2., 252.,   0., ...,   0.,   0.,   0.])
            in_bbox = np.zeros((n_category, len(sp_visible_pts)), dtype=bool)
            if len(sp_visible_pts) != 0:
                sp_coor = screen_coor[sp_visible_pts]
                bb1 = {'x1': sp_coor[:, 0].min(), 'y1': sp_coor[:, 1].min(), \
                        'x2': sp_coor[:, 0].max(), 'y2': sp_coor[:, 1].max()}
            for pred in valid_preds:
                cat_id = pred["category_id"]
        
                mask_points = np.nonzero(pred["mask"])
                x1, x2 = np.asarray(mask_points[1]).min(), np.asarray(mask_points[1]).max()
                y1, y2 = np.asarray(mask_points[0]).min(), np.asarray(mask_points[0]).max()
                
                if len(sp_visible_pts) == 0:
                    in_box_ratio_list[cat_id][k].append(-1)
                else:
                    bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    #Eg: {'x1': 285.15826416015625, 'y1': 151.8497314453125, 'x2': 516.8762817382812, 'y2': 424.0416259765625}
                    if get_iou(bb1, bb2) < 1e-6:
                        in_box_ratio_list[cat_id][k].append(0)
                    else:
                        #  mask= check_pc_within_bbox(x1, y1, x2, y2, sp_coor)
                        mask = pred["mask"]
                        point_seg_mask = check_pc_within_seg_mask(pred["mask"], np.int16(sp_coor))
                        in_bbox[cat_id] = np.logical_or(in_bbox[cat_id], point_seg_mask) 
                       
                        in_box_ratio_list[cat_id][k].append(point_seg_mask.mean())
             
            for j in range(n_category):
                sp_bbox_visible_cnt[j, k] += in_bbox[j].sum() 
 
    
    sem_score = sp_bbox_visible_cnt / (sp_visible_cnt.reshape(1, -1) + 1e-6)
    sem_score[:, sp_visible_cnt == 0] = 0
    sem_seg = np.ones(xyz.shape[0], dtype=np.int32) * -1

    # assign semantic labels to superpoints
    for i in range(n_sp):
        if sem_score[:, i].max() < 0.5:
            continue
        idx = -1
        for j in reversed(range(n_category)): #give priority to small parts
            if sem_score[j, i] >= 0.5 and part_names[j] in ["handle", "button", "wheel", "knob", "switch", "bulb", "shaft", "touchpad", "camera", "screw"]:
                idx = j
                break
        if idx == -1:
            idx = np.argmax(sem_score[:, i])
        sem_seg[superpoint[i]] = idx
   
    if visualize:
        print("visualizing...")
        os.makedirs("%s/semantic_seg" % save_dir, exist_ok=True)  
        cmap = plt.get_cmap('tab10')  # You can choose a different colormap if needed
        colors = [cmap(i)[:3] for i in range(n_category)]  # Extract RGB components
        unique_colors = ListedColormap(colors)
        for j in range(n_category):
            mask = (sem_seg == j).reshape(-1, 1)
            rgb_sem = np.ones((xyz.shape[0], 3)) * np.tile(colors[j], (xyz.shape[0], 1)) * mask
            # rgb_sem = np.ones((xyz.shape[0], 3)) * (sem_seg == j).reshape(-1, 1) * colors[j]
            save_colored_pc("%s/semantic_seg/%s.ply" % (save_dir, part_names[j]), xyz, rgb_sem)
    # os.system(f"aws s3 cp {save_dir} s3://mbz-hpc-aws-master/AROARU6TOWKRU3FNVE2PB:Amrin.Kareem@mbzuai.ac.ae/PartLISA/{save_dir} --region me-central-1")

        return sem_seg, None