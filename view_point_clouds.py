import open3d as o3d
import argparse
import glob
import os
import time
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Sequence Viewer")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .ply files")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second")
    parser.add_argument("--no_cache", action="store_true", help="Disable binary caching")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.ply")))
    
    # Filter out cached files if they are in the same dir (unlikely if we put them in subdir)
    # But just in case user runs on cache dir
    files = [f for f in files if "cached_binary" not in f]

    if not files:
        print(f"No .ply files found in {args.dir}")
        return

    # Cache handling
    ply_files_to_load = files
    if not args.no_cache:
        cache_dir = os.path.join(args.dir, "cached_binary_opt")
        os.makedirs(cache_dir, exist_ok=True)
        
        cached_files = []
        print("Checking/Creating optimized binary cache...")
        print("Steps: 1. Convert to Binary 2. Fix coords (CV->GLTF) 3. Center 4. Downsample")
        
        needs_convert = []
        for f in files:
            bn = os.path.basename(f)
            cache_path = os.path.join(cache_dir, bn)
            if not os.path.exists(cache_path):
                needs_convert.append((f, cache_path))
            cached_files.append(cache_path)
            
        if needs_convert:
            print(f"Processing {len(needs_convert)} files...")
            
            # Calculate global center from the FIRST file in the sequence (to maintain relative motion)
            # We must load it to get the center.
            first_file_src = files[0] # Orig RGB/Depth
            pcd_first = o3d.io.read_point_cloud(first_file_src)
            
            # 1. Flip Y/Z (CV -> GLTF)
            # CV: Y Down, Z Forward. GLTF: Y Up, Z Backward.
            points = np.asarray(pcd_first.points)
            points[:, 1] *= -1
            points[:, 2] *= -1
            pcd_first.points = o3d.utility.Vector3dVector(points)
            
            # 2. Get Center
            center = pcd_first.get_center()
            print(f"First frame center: {center}")
            
            del pcd_first
            del points
            
            for src, dst in tqdm(needs_convert):
                pcd = o3d.io.read_point_cloud(src)
                
                # 1. Flip Y/Z
                pts = np.asarray(pcd.points)
                pts[:, 1] *= -1
                pts[:, 2] *= -1
                
                # 2. Quantize/Downsample first? No, center first.
                pts -= center
                
                pcd.points = o3d.utility.Vector3dVector(pts)
                
                # 3. Downsample
                # Target points ~500k-1M for smooth playback on average GPU
                # If points > 1M, downsample.
                n_pts = len(pts)
                if n_pts > 1_000_000:
                    # Random downsample
                    # ratio = 1_000_000 / n_pts
                    # pcd = pcd.random_down_sample(ratio) 
                    # Voxel downsample is better/uniform but slower?
                    # Let's use uniform downsample - fast.
                    every_k_points = int(n_pts / 1_000_000)
                    if every_k_points > 1:
                         pcd = pcd.uniform_down_sample(every_k_points)
                
                # Write binary
                o3d.io.write_point_cloud(dst, pcd, write_ascii=False)
        else:
            print("All files cached (Optimized).")
            
        ply_files_to_load = cached_files

    print(f"Found {len(ply_files_to_load)} files. Press 'Q' to quit, 'SPACE' to pause.")

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Player", width=1280, height=720)

    # State
    class State:
        index = 0
        playing = True
        geometry = None

    state = State()

    # Load first geometry
    pcd = o3d.io.read_point_cloud(ply_files_to_load[0])
    vis.add_geometry(pcd)
    state.geometry = pcd

    # View control setup
    ctr = vis.get_view_control()
    # Optional: Set better initial view if needed
    
    # Key Callbacks
    def toggle_play(vis):
        state.playing = not state.playing
        print(f"Paused: {not state.playing}")
        return False
        
    vis.register_key_callback(32, toggle_play) # Space
    vis.register_key_callback(81, lambda v: v.close()) # Q

    # Animation Loop
    last_time = time.time()
    frame_time = 1.0 / args.fps

    while True:
        current_time = time.time()
        
        # If playing or just starting (to ensure first frame render?)
        if state.playing and (current_time - last_time >= frame_time):
            state.index = (state.index + 1) % len(ply_files_to_load)
            
            # Load next
            # Open3D binary load is fast (~ms for 1M points)
            new_pcd = o3d.io.read_point_cloud(ply_files_to_load[state.index])
            
            # Swap
            # Note: Preserving view_control is automatic if we don't destroy window
            # But removing/adding geometry might reset view if reset_bounding_box=True (default)
            # We explicitly say False.
            vis.remove_geometry(state.geometry, reset_bounding_box=False)
            vis.add_geometry(new_pcd, reset_bounding_box=False)
            state.geometry = new_pcd
            
            last_time = current_time

        if not vis.poll_events():
            break
            
        vis.update_renderer()
        
        # Small sleep to prevent 100% CPU usage if FPS is low
        # e.g. if we have 90ms left, sleep a bit
        elapsed = time.time() - last_time
        if state.playing and elapsed < frame_time:
             time.sleep(max(0, (frame_time - elapsed) * 0.9))

    vis.destroy_window()

if __name__ == "__main__":
    main()
