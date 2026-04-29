import numpy as np
from pathlib import Path
from decord import VideoReader, cpu

# Input dataset root
input_root = Path("data/ek55_processed/train")

frames_num = 16
original_fps = 4
desired_fps = 0.5

# Output dataset root
output_root = Path(f"data/ek55_{frames_num }frames_{desired_fps}fps_train")

import cv2
import os

def process_video(video_path, output_base_dir):
    try:
        vr = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
    except Exception as e:
        print(f"Failed to load {video_path}: {e}")
        return
    
    fstp = original_fps // desired_fps # frames to skip
    clip_len = int(frames_num * fstp)

    # Create output directory for this video
    output_base_dir.mkdir(parents=True, exist_ok=True)
    base_fname = video_path.stem + "_{0:03d}.mp4"

    # Skipping video if it's too short
    if len(vr) < clip_len:
        print(f"skipping video of length {len(vr)}")
        return [], None


    for i in range(0, len(vr), clip_len):
        indices = np.linspace(i, i + clip_len, num=frames_num, endpoint=False).astype(np.int64)

        if indices[-1] < len(vr):
            fname = base_fname.format(i // clip_len)
            save_path = output_base_dir / fname

            if save_path.exists():
                continue

            try:
                frames = vr.get_batch(indices)
            except Exception as e:
                print(f"Error reading frames in {video_path}: {e}")
                continue

            if frames.shape[0] != frames_num:
                print(f"Frame mismatch in {video_path}")
                continue


            # np.save(save_path, frames.asnumpy())
            # Export the video
            T, H, W, C = frames.shape
            # video = frames.asnumpy().tobytes()
            video = frames.asnumpy()
            
            out = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                desired_fps,
                (W, H)
            )
            for frame in video:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()

for video_path in input_root.rglob("*.mp4"):

    print(f"Processing: {video_path}")
    process_video(video_path, output_root)

print("Done.")