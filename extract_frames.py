import cv2
import os

def extract_frames(video_path, output_dir):
    print(f"Starting extraction for {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to extract: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        filename = f"frame_{frame_count:06d}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)
        
        if frame_count % 100 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames saved to {output_dir}", end='\r')
        
        frame_count += 1
    
    cap.release()
    print(f"\nFinished extracting {frame_count} frames from {video_path} to {output_dir}")

def main():
    # Define paths
    video_left = os.path.join("assets", "GX011934.MP4")
    output_left = os.path.join("assets", "frames_left")
    
    video_right = os.path.join("assets", "GX011920.MP4")
    output_right = os.path.join("assets", "frames_right")

    # Run extraction
    extract_frames(video_left, output_left)
    extract_frames(video_right, output_right)

if __name__ == "__main__":
    main()
