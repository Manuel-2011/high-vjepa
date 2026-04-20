import subprocess
from pathlib import Path

# Input dataset root
input_root = Path("/mnt/disk/2g1n6qdydwa9u22shpxqzp0t8m")  # contains train/ and test/

# Output dataset root
output_root = Path("/mnt/disk/ek100-processed")

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


    
for video_path in input_root.rglob("*.MP4"):
    # Define output path
    output_path = output_root / video_path.name
    
    # Change filename
    output_path = output_path.with_name(
        video_path.stem + "_4fps_720p.mp4"
    )
    
    print(f"Processing: {video_path} -> {output_path}")
    process_video(video_path, output_path)

print("Done.")