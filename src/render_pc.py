from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)
from pytorch3d.structures import Pointclouds
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def render_single_view(pc, view, device, background_color=(1,1,1), resolution=800, camera_distance=2.2, 
                        point_size=0.005, points_per_pixel=1, bin_size=0, znear=0.01):
    R, T = look_at_view_transform(camera_distance, view[0], view[1])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear) 
    #calculates the camera transformation R (rotation matrix) and T (translation vector) based on the given camera distance and view angles. 
    #It then creates a perspective camera (cameras) with the specified transformation and near clipping plane.

    raster_settings = PointsRasterizationSettings(
        image_size=resolution, 
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    compositor=NormWeightedCompositor(background_color=background_color)
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=compositor
    )
    img = renderer(pc)
    pc_idx = rasterizer(pc).idx
    screen_coords = cameras.transform_points_screen(pc._points_list[0], image_size=(resolution, resolution))
    return img, pc_idx, screen_coords
    
def render_pc(xyz, rgb, save_dir, device):
    
    pc = Pointclouds(points=[torch.Tensor(xyz).to(device)],
                    features=[torch.Tensor(rgb).to(device)])
    #pc = io.load_pointcloud(pc_file, device=device)
    
    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)

    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    pc_idx_list = []
    screen_coords_list = []

    for i, view in enumerate(views):
        img, pc_idx, screen_coords = render_single_view(pc, view, device)
        plt.imsave(os.path.join(img_dir, f"{i}.png"), img[0, ..., :3].cpu().numpy() * 0.99999)
        pc_idx_list.append(pc_idx)
        screen_coords_list.append(screen_coords)

    pc_idx = torch.cat(pc_idx_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]
    np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())

    return img_dir, pc_idx.cpu().numpy(), screen_coords.cpu().numpy()
