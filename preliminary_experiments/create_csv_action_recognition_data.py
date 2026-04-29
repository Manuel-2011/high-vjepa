import csv
import argparse
from pathlib import Path


def convert_csv(input_csv, output_csv, base_path):
    """
    Reads the input CSV and writes a new CSV with:
    - No headers
    - First column: base_path + narration_id + ".mp4"
    - Second column: "verb_class,noun_class"
    """

    with open(input_csv, mode="r", newline="", encoding="utf-8") as infile, \
         open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile, delimiter=" ")

        for row in reader:
            video_id = row["video_id"]
            narration_id = row["narration_id"]
            verb_class = row["verb_class"]
            noun_class = row["noun_class"]

            video_path = f"{base_path}{video_id}_{narration_id}.mp4"
            labels = f"{verb_class},{noun_class}"

            # Write without headers:
            # first column = video path
            # second column = labels
            writer.writerow([video_path, labels])


def main():
    parser = argparse.ArgumentParser(
        description="Convert EPIC-KITCHENS style CSV into label CSV."
    )

    parser.add_argument(
        "input_csv",
        help="Path to input CSV file"
    )

    parser.add_argument(
        "output_csv",
        help="Path to output CSV file"
    )

    parser.add_argument(
        "--base_path",
        default="./data/",
        help="Constant prefix for video path (default: /your/constant/path/)"
    )

    args = parser.parse_args()

    convert_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        base_path=args.base_path
    )

    print(f"Output written to: {args.output_csv}")


if __name__ == "__main__":
    main()