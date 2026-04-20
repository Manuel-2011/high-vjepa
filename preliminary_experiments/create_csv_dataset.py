import os
import csv
from pathlib import Path


def generate_video_csv(folder_path: str):
    # Resolve paths
    folder = Path(folder_path).resolve()
    project_root = Path.cwd().resolve()

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder}")

    # Prepare output directory
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Output file name = folder name
    output_file = data_dir / f"{folder.name}.csv"

    rows = []

    # Walk through folder recursively
    for file_path in folder.rglob("*.mp4"):
        # Compute relative path to project root
        relative_path = file_path.resolve().relative_to(project_root)
        rows.append([str(relative_path), "no-label"])

    # Write CSV (space-separated)
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(rows)

    print(f"CSV saved to: {output_file}")
    print(f"Total files found: {len(rows)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CSV from mp4 files.")
    parser.add_argument("folder", help="Path to folder containing mp4 files")

    args = parser.parse_args()
    generate_video_csv(args.folder)