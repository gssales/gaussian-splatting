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

import os
from argparse import ArgumentParser
import time

# ref_real_scenes = ["ref_real/gardenspheres", "ref_real/sedan", "ref_real/toycar"]
# refnerf_scenes = ["shiny_blender/ball","shiny_blender/car","shiny_blender/coffee","shiny_blender/helmet","shiny_blender/teapot","shiny_blender/toaster"]
# nerf_synthetic_scenes = ["nerf_synthetic/chair","nerf_synthetic/drums","nerf_synthetic/ficus","nerf_synthetic/hotdog","nerf_synthetic/lego","nerf_synthetic/materials","nerf_synthetic/mic","nerf_synthetic/ship"]
# glossy_synthetic_scenes = ["GlossySynthetic/angel","GlossySynthetic/bell","GlossySynthetic/cat","GlossySynthetic/horse","GlossySynthetic/luyu","GlossySynthetic/potion","GlossySynthetic/tbell","GlossySynthetic/teapot"]

ref_real_scenes = []
refnerf_scenes = []
nerf_synthetic_scenes = []
glossy_synthetic_scenes = ["GlossySynthetic/angel","GlossySynthetic/bell","GlossySynthetic/cat","GlossySynthetic/horse","GlossySynthetic/luyu","GlossySynthetic/potion","GlossySynthetic/tbell","GlossySynthetic/teapot"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/3dgs/eval")


extra_args = {
  "ref_real/sedan": " -r 8",
  "ref_real/gardenspheres": " -r 4",
  "ref_real/toycar": " -r 4",
}

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(ref_real_scenes)
all_scenes.extend(refnerf_scenes)
all_scenes.extend(nerf_synthetic_scenes)
all_scenes.extend(glossy_synthetic_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--ref_real', type=str, default="/mnt/data")
    parser.add_argument('--refnerf', type=str, default="/mnt/data")
    parser.add_argument('--nerf_synthetic', type=str, default="/mnt/data")
    parser.add_argument('--glossy_synthetic', type=str, default="/mnt/data")
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 --save_iterations 7000 30000"
    
    start_time = time.time()
    for scene in ref_real_scenes:
        source = args.ref_real + "/" + scene
        extra = extra_args[scene]
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
    ref_real_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in refnerf_scenes:
        source = args.refnerf + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    refnerf_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in nerf_synthetic_scenes:
        source = args.nerf_synthetic + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    nerf_synthetic_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in glossy_synthetic_scenes:
        source = args.glossy_synthetic + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    glossy_synthetic_timing = (time.time() - start_time)/60.0

    with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
        file.write(f"ref_real: {ref_real_timing} minutes \n shiny_blender: {refnerf_timing} minutes \n nerf_synthetic: {nerf_synthetic_timing} minutes \n GlossySynthetic: {glossy_synthetic_timing} minutes \n")

if not args.skip_rendering:
    all_sources = []
    for scene in ref_real_scenes:
        all_sources.append(args.ref_real + "/" + scene)
    for scene in refnerf_scenes:
        all_sources.append(args.refnerf + "/" + scene)
    for scene in nerf_synthetic_scenes:
        all_sources.append(args.nerf_synthetic + "/" + scene)
    for scene in glossy_synthetic_scenes:
        all_sources.append(args.glossy_synthetic + "/" + scene)
    
    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
