import csv
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def timestamp_to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def extract_clip(input_video, output_video, start, end, reencode=False):
    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264",
            "-c:a", "aac",
            output_video
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_video,
            "-c", "copy",
            output_video
        ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def resolve_video_path(video_id, videos_dirs, videos_extensions):
    for videos_dir, extension in zip(videos_dirs, videos_extensions):
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    return None

def process_row(row, videos_dirs, videos_extensions, output_dir, reencode):
    video_id = row["video_id"]
    start_ts = row["start_timestamp"]
    end_ts = row["stop_timestamp"]
    narration_id = row["narration_id"]

    input_video = resolve_video_path(video_id, videos_dirs, videos_extensions)

    if input_video is None:
        return ("missing", video_id)

    output_name = f"{video_id}_{narration_id}.mp4"
    output_path = output_dir / output_name

    if output_path.exists():
        return ("skipped", str(output_path))

    start_sec = timestamp_to_seconds(start_ts)
    end_sec = timestamp_to_seconds(end_ts)

    success = extract_clip(
        str(input_video),
        str(output_path),
        start_sec,
        end_sec,
        reencode=reencode
    )

    if success:
        return ("saved", str(output_path))
    else:
        return ("failed", str(output_path))

def process_csv(csv_path, videos_dirs, videos_extensions, output_dir,
                reencode=False, max_workers=None):

    videos_dirs = [Path(p) for p in videos_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        # Good default: avoid oversubscribing CPU heavily
        max_workers = min(8, os.cpu_count() or 4)

    tasks = []
    missing_count = 0

    # Load all rows first (important for parallel dispatch)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_row,
                row,
                videos_dirs,
                videos_extensions,
                output_dir,
                reencode
            )
            for row in reader
        ]

        for future in as_completed(futures):
            status, info = future.result()

            if status == "missing":
                print(f"Missing video: {info}")
                missing_count += 1
            # elif status == "saved":
            #     print(f"Saved: {info}")
            elif status == "failed":
                print(f"Failed: {info}")
            # skipped → silent (like your original behavior)

    print("missing segments", missing_count)

if __name__ == "__main__":
    process_csv(
        csv_path="data/EPIC_100_validation.csv",
        videos_dirs=[
            "data/ek55_processed/train",
            "data/ek100-processed",
            "data/ek55_processed/test"
        ],
        videos_extensions=[
            "_4fps_720p.mp4",
            "_4ps_720p.mp4",
            "_4fps_720p.mp4"
        ],
        output_dir="data/ek100_action_clips_validation",
        reencode=True,
        max_workers=1  # tune this
    )