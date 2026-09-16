#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os, time
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
from typing import Tuple
import copy
import mediapy as media
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from functools import partial

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

  @torch.no_grad()
  def render_export(self, viewpoint_stack, path):
    render_path = os.path.join(path, "renders")
    os.makedirs(render_path, exist_ok=True)
    self.viewpoint_stack = viewpoint_stack
    for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="render + export images"):
      render_pkg = self.render(viewpoint_cam, self.gaussians)
      rgb = render_pkg['render']
      save_img_u8(rgb.permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]

def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform

def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])

def _fit_horizontal_ellipse(poses: np.ndarray,
                            center: np.ndarray):
  """Fit an ellipse centered at ``center`` using all camera positions.

  The poses are expected to have been PCA-aligned first, so the first two
  coordinates span the dominant camera-motion plane.  We fit the symmetric
  quadratic form d.T @ Q @ d = 1.  Its eigenvectors give the ellipse
  directions and its eigenvalues give the two radii.
  """
  xy = poses[:, :2, 3] - center[None, :2]
  design = np.stack([xy[:, 0] ** 2,
                     2.0 * xy[:, 0] * xy[:, 1],
                     xy[:, 1] ** 2], axis=-1)
  q, _, _, _ = np.linalg.lstsq(design, np.ones(len(xy)), rcond=None)
  quadratic = np.array([[q[0], q[1]], [q[1], q[2]]])

  # Numerical noise (or a very short arc) can make the unconstrained fit
  # slightly indefinite.  Clamp it to the nearest usable positive form.
  eigenvalues, axes = np.linalg.eigh(quadratic)
  scale = max(np.max(np.abs(eigenvalues)), 1e-8)
  eigenvalues = np.maximum(eigenvalues, scale * 1e-6)
  radii = 1.0 / np.sqrt(eigenvalues)
  return axes, radii


def _smallest_angle_interval(angles: np.ndarray):
  """Return the unwrapped interval containing a set of circular angles."""
  wrapped = np.sort(np.mod(angles, 2.0 * np.pi))
  gaps = np.diff(np.concatenate([wrapped, wrapped[:1] + 2.0 * np.pi]))
  start_index = (np.argmax(gaps) + 1) % len(wrapped)
  ordered = np.concatenate([wrapped[start_index:],
                            wrapped[:start_index] + 2.0 * np.pi])
  return ordered[0], ordered[-1]


def generate_arc_figure8_path(poses: np.ndarray,
                              n_frames: int = 480,
                              height_amplitude: float = None,
                              arc_scale: float = 1.0) -> np.ndarray:
  """Generate a closed vertical figure-eight over the observed camera arc.

  The horizontal ellipse and the occupied angular interval are estimated from
  all input poses.  During one loop, the camera traverses the arc forward and
  backward.  Its height follows sin(2t), producing a figure eight in the
  arc-length/height plane.

  Args:
    poses: (N, 3, 4) camera-to-world poses in a PCA-aligned coordinate frame.
    n_frames: Number of output poses.  The duplicated final pose is omitted.
    height_amplitude: Vertical displacement from the median capture height.
      If None, it is inferred from the input height range.
    arc_scale: Multiplier for the observed angular extent about its midpoint.
  """
  if len(poses) < 3:
    raise ValueError("At least three viewpoints are required to fit an arc.")

  center = focus_point_fn(poses)
  axes, radii = _fit_horizontal_ellipse(poses, center)

  local_xy = (poses[:, :2, 3] - center[None, :2]) @ axes
  unit_xy = local_xy / radii[None, :]
  observed_angles = np.arctan2(unit_xy[:, 1], unit_xy[:, 0])
  angle_start, angle_end = _smallest_angle_interval(observed_angles)
  angle_mid = 0.5 * (angle_start + angle_end)
  angle_half_range = 0.5 * (angle_end - angle_start) * arc_scale

  heights = poses[:, 2, 3]
  base_height = np.median(heights)
  if height_amplitude is None:
    observed_height = np.percentile(heights, 90) - np.percentile(heights, 10)
    # A nearly planar capture still needs a visible vertical motion.
    height_amplitude = max(0.5 * observed_height, 0.08 * np.mean(radii))

  phase = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
  # cos(phase) moves from one arc endpoint to the other and back.
  arc_angles = angle_mid + angle_half_range * np.cos(phase)
  ellipse_local = np.stack([radii[0] * np.cos(arc_angles),
                            radii[1] * np.sin(arc_angles)], axis=-1)
  positions = np.empty((n_frames, 3), dtype=poses.dtype)
  positions[:, :2] = center[None, :2] + ellipse_local @ axes.T
  positions[:, 2] = base_height + height_amplitude * np.sin(2.0 * phase)

  avg_up = normalize(poses[:, :3, 1].mean(0))
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
  return np.stack([viewmatrix(p - center, up, p) for p in positions])

def generate_path(viewpoint_cameras, n_frames=480):
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  # generate new poses
  new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
  # warp back to orignal scale
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []
  for c2w in new_poses:
      c2w = c2w @ np.diag([1, -1, -1, 1])
      cam = copy.deepcopy(viewpoint_cameras[0])
      cam.image_height = int(cam.image_height / 2) * 2
      cam.image_width = int(cam.image_width / 2) * 2
      cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
      cam.camera_center = cam.world_view_transform.inverse()[3, :3]
      traj.append(cam)

  return traj


def generate_arc_figure8_camera_path(viewpoint_cameras,
                                     n_frames=480,
                                     height_amplitude=None,
                                     arc_scale=1.0):
  """Camera-object wrapper for :func:`generate_arc_figure8_path`."""
  c2ws = np.array([
      np.linalg.inv(np.asarray(cam.world_view_transform.T.cpu().numpy()))
      for cam in viewpoint_cameras
  ])
  poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
  poses_recentered, colmap_to_world_transform = transform_poses_pca(poses)

  new_poses = generate_arc_figure8_path(
      poses_recentered,
      n_frames=n_frames,
      height_amplitude=height_amplitude,
      arc_scale=arc_scale,
  )
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []
  for c2w in new_poses:
    c2w = c2w @ np.diag([1, -1, -1, 1])
    cam = copy.deepcopy(viewpoint_cameras[0])
    cam.image_height = int(cam.image_height / 2) * 2
    cam.image_width = int(cam.image_width / 2) * 2
    cam.world_view_transform = torch.from_numpy(
        np.linalg.inv(c2w).T).float().cuda()
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0)
        .bmm(cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    traj.append(cam)

  return traj

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image

def create_videos(base_dir, input_dir, out_name, num_frames=480):
  """Creates videos out of the images saved to disk."""
  # Last two parts of checkpoint path are experiment name and scene name.
  video_prefix = f'{out_name}'

  zpad = max(5, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  os.makedirs(base_dir, exist_ok=True)
  render_dist_curve_fn = np.log
  
  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.png')
  depth_frame = load_img(depth_file)
  shape = depth_frame.shape
  p = 3
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')

  video_kwargs = {
      'shape': shape[:2],
      'codec': 'h264',
      'fps': 60,
      'crf': 18,
  }
  
  for k in ['normal', 'color']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
    input_format = 'gray' if k == 'alpha' else 'rgb'
    

    file_ext = 'png' if k in ['color', 'normal'] else 'tiff'
    idx = 0

    if k == 'color':
      file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
    else:
      file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

    if not os.path.exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in tqdm(range(num_frames)):
        # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        if k == 'color':
          img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
        else:
          img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

        if not os.path.exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = load_img(img_file)
        if k in ['color', 'normal']:
          img = img / 255.
        elif k.startswith('depth'):
          img = render_dist_curve_fn(img)
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          img = cm.get_cmap('turbo')(img)[..., :3]

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)
        idx += 1

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--path", default="ellipse", type=str)
    parser.add_argument("--frames", default=480, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    

    train_cams = scene.getTrainCameras()
    for id in range(len(train_cams)):
      for attribute in [
        "original_image",
        "alpha_mask",
        "depth",
        "invdepthmap",
      ]:
        if hasattr(train_cams[id], attribute):
          setattr(train_cams[id], attribute, None)
    
    print("render videos ...")
    traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
    os.makedirs(traj_dir, exist_ok=True)
    n_frames = args.frames
    if args.path == "ellipse":
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
    elif args.path == "arc":
      cam_traj = generate_arc_figure8_camera_path(scene.getTrainCameras(), n_frames=n_frames, arc_scale=0.7)
    gaussExtractor.render_export(cam_traj, traj_dir)
    create_videos(base_dir=traj_dir,
                input_dir=traj_dir, 
                out_name='render_traj', 
                num_frames=n_frames)
