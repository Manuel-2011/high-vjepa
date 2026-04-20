import os
import cv2

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cap.release()
    
    if fps == 0:
        return None
    
    duration = frame_count / fps
    return duration


def process_dataset(root_dir):
    results = {}

    for split in ['train', 'test']:
        split_path = os.path.join(root_dir, split)
        video_durations = []
        videos_no_duration = 0

        if not os.path.exists(split_path):
            continue

        for person in os.listdir(split_path):
            person_path = os.path.join(split_path, person)

            if not os.path.isdir(person_path):
                continue

            for file in os.listdir(person_path):
                if file.lower().endswith('.mp4'):
                    video_path = os.path.join(person_path, file)
                    duration = get_video_duration(video_path)

                    if duration is not None:
                        video_durations.append((video_path, duration))
                    else:
                        videos_no_duration += 1

        total_videos = len(video_durations)
        avg_duration = (
            sum(d for _, d in video_durations) / total_videos
            if total_videos > 0 else 0
        )

        results[split] = {
            'total_videos': total_videos,
            'average_duration': avg_duration,
            'videos': video_durations,
            'videos_without_duration': videos_no_duration
        }

    return results


def write_report(results, output_file):
    with open(output_file, 'w') as f:
        for split, data in results.items():
            f.write(f"=== {split.upper()} DATASET ===\n")
            f.write(f"Total videos: {data['total_videos'] + data['videos_without_duration']}\n")
            f.write(f"Videos without duration information: {data['videos_without_duration']}\n")
            f.write(f"Average duration (seconds): {data['average_duration']:.2f}\n\n")

            f.write("Individual video durations:\n")
            for path, duration in data['videos']:
                f.write(f"{path} -> {duration:.2f} sec\n")

            f.write("\n" + "="*40 + "\n\n")


if __name__ == "__main__":
    root_directory = "/mnt/disk/3h91syskeag572hl6tvuovwv4d/videos"  # <-- change this
    output_txt = "dataset_report.txt"

    results = process_dataset(root_directory)
    write_report(results, output_txt)

    print(f"Report saved to {output_txt}")