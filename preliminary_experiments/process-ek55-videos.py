import subprocess
from pathlib import Path

# Input dataset root
input_root = Path("/mnt/disk/3h91syskeag572hl6tvuovwv4d/videos")  # contains train/ and test/

# Output dataset root
output_root = Path("/mnt/disk/ek55_processed")

# ffmpeg command template
def process_video(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-threads", "4",
        "-i", str(input_path),
        "-vf", "fps=4,scale=-2:720",
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# Traverse both train and test folders
for split in ["train", "test"]:
    input_dir = input_root / split
    
    for video_path in input_dir.rglob("*.MP4"):
        # Define output path
        output_path = output_root / split
        
        # Change filename (optional, matches your example)
        output_path = output_path / (video_path.stem + "_4fps_720p.mp4")
        
        print(f"Processing: {video_path} -> {output_path}")
        process_video(video_path, output_path)

print("Done.")