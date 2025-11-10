#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
# import open3d as o3d
# import trimesh

class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            self.rgbmaps.append(rgb.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)

    @torch.no_grad()
    def export_image(self, path, export_gt=True):
        render_path = os.path.join(path, "renders")
        if export_gt:
            gts_path = os.path.join(path, "gt")
            os.makedirs(gts_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            if export_gt:
                gt = viewpoint_cam.original_image[0:3, :, :]
                save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
