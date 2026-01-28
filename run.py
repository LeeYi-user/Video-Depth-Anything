# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import cv2
import json
import numpy as np
import os
import torch

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')
    
    # Keyframe inference arguments
    parser.add_argument('--key_frame', type=str, help='Keyframe filename (e.g., frame_015314)')
    parser.add_argument('--fps', type=float, default=29.97, help='FPS of the video')
    parser.add_argument('--right_offset', type=int, default=81, help='Offset for the right camera frames')
    parser.add_argument('--duration', type=float, default=2.0, help='Duration in seconds (one side) for inference')
    parser.add_argument('--left_dir', type=str, default='assets/frames_left', help='Directory for left frames')
    parser.add_argument('--right_dir', type=str, default='assets/frames_right', help='Directory for right frames')
    parser.add_argument('--run_keyframe_mode', action='store_true', help='Run in keyframe inference mode')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    if args.run_keyframe_mode:
        if not args.key_frame:
            print("Error: --key_frame is required for keyframe mode.")
            exit(1)
        
        # Parse keyframe index
        try:
            key_idx = int(args.key_frame.split('_')[-1].split('.')[0])
        except ValueError:
            print(f"Error: Could not parse frame index from {args.key_frame}")
            exit(1)
            
        # Calculate range
        range_frames = int(args.fps * args.duration)
        start_idx = max(0, key_idx - range_frames)
        end_idx = key_idx + range_frames
        
        # Function to load frames
        def load_frames(directory, start, end, indices_map):
            loaded_frames = []
            valid_indices = []
            
            # Get list of all files to ensure we match correctly or just construct filenames
            # Assuming format "frame_XXXXXX.jpg" as per extract_frames.py
            
            for idx in range(start, end + 1):
                filename = f"frame_{idx:06d}.jpg"
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        loaded_frames.append(img)
                        valid_indices.append(idx)
                        indices_map.append(idx)
                    else:
                        print(f"Warning: Could not read {filepath}")
                else:
                    # simplistic handling: skip
                    pass
            
            return np.stack(loaded_frames, axis=0) if loaded_frames else None

        # Load Left Frames
        left_indices = []
        print(f"Loading left frames from {start_idx} to {end_idx}...")
        frames_left = load_frames(args.left_dir, start_idx, end_idx, left_indices)
        
        if frames_left is None:
            print("Error: No left frames loaded.")
            exit(1)
            
        # Load Right Frames
        right_start_idx = start_idx + args.right_offset
        right_end_idx = end_idx + args.right_offset
        right_indices = []
        print(f"Loading right frames from {right_start_idx} to {right_end_idx}...")
        frames_right = load_frames(args.right_dir, right_start_idx, right_end_idx, right_indices)
        
        if frames_right is None:
            print("Error: No right frames loaded.")
            exit(1)
            
        # Inference
        print("Running inference on Left frames...")
        depths_left, _ = video_depth_anything.infer_video_depth(frames_left, args.fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
        
        print("Running inference on Right frames...")
        depths_right, _ = video_depth_anything.infer_video_depth(frames_right, args.fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
        
        # Find keyframe index in the loaded chunks
        try:
            left_local_idx = left_indices.index(key_idx)
            right_key_idx = key_idx + args.right_offset
            right_local_idx = right_indices.index(right_key_idx)
        except ValueError:
            print("Error: Keyframe not found in the loaded sequence (this shouldn't happen logic-wise if files exist).")
            exit(1)
            
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save args
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Save Left
        left_key_depth = depths_left[left_local_idx]
        left_key_name = f"frame_{key_idx:06d}"
        
        # Save vis
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        def save_vis(depth, path):
            d_min, d_max = depth.min(), depth.max()
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            colormap = np.array(cm.get_cmap("inferno").colors)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
            cv2.imwrite(path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

        save_vis(left_key_depth, os.path.join(args.output_dir, f"{left_key_name}_vis.png"))
        np.save(os.path.join(args.output_dir, f"{left_key_name}.npy"), left_key_depth)
        
        # Save Left Video
        save_video(frames_left, os.path.join(args.output_dir, f"{left_key_name}_left_src.mp4"), fps=args.fps)
        save_video(depths_left, os.path.join(args.output_dir, f"{left_key_name}_left_vis.mp4"), fps=args.fps, is_depths=True, grayscale=args.grayscale)
        
        print(f"Saved Left results to {args.output_dir}/{left_key_name}...")
        
        # Save Right
        right_key_depth = depths_right[right_local_idx]
        right_key_name = f"frame_{right_key_idx:06d}"
        
        save_vis(right_key_depth, os.path.join(args.output_dir, f"{right_key_name}_vis.png"))
        np.save(os.path.join(args.output_dir, f"{right_key_name}.npy"), right_key_depth)
        
        # Save Right Video
        save_video(frames_right, os.path.join(args.output_dir, f"{right_key_name}_right_src.mp4"), fps=args.fps)
        save_video(depths_right, os.path.join(args.output_dir, f"{right_key_name}_right_vis.mp4"), fps=args.fps, is_depths=True, grayscale=args.grayscale)
        
        print(f"Saved Right results to {args.output_dir}/{right_key_name}...")
        
    else:
        # Original Logic
        frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        video_name = os.path.basename(args.input_video)
        os.makedirs(args.output_dir, exist_ok=True)

        processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
        save_video(frames, processed_video_path, fps=fps)
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

        if args.save_npz:
            depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
            np.savez_compressed(depth_npz_path, depths=depths)
        if args.save_exr:
            depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
            os.makedirs(depth_exr_dir, exist_ok=True)
            import OpenEXR
            import Imath
            for i, depth in enumerate(depths):
                output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": depth.tobytes()})
                exr_file.close()

        if args.metric:
            import open3d as o3d

            width, height = depths[0].shape[-1], depths[0].shape[-2]
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / args.focal_length_x
            y = (y - height / 2) / args.focal_length_y

            for i, (color_image, depth) in enumerate(zip(frames, depths)):
                z = np.array(depth)
                points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                colors = np.array(color_image).reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(os.path.join(args.output_dir, 'point' + str(i).zfill(4) + '.ply'), pcd)
