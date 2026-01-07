import os
import glob
from tqdm import tqdm

def parse_line(line):
    parts = line.strip().split(' ')
    if len(parts) < 3:
        return None, None, None
    
    # Handle paths with spaces
    path = " ".join(parts[:-2])
    frame_count = int(parts[-2])
    label = int(parts[-1])
    return path, frame_count, label

def verify_annotations(annotation_file_path):
    print(f"\n--- Verifying Annotation File: {annotation_file_path} ---")

    if not os.path.isfile(annotation_file_path):
        print(f"ERROR: Annotation file not found!")
        return

    with open(annotation_file_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("ERROR: Annotation file is empty!")
        return

    total_lines = len(lines)
    valid_lines = 0
    mismatch_count = 0
    path_not_found = 0
    
    print(f"Found {total_lines} lines to verify. Checking each line...")

    for line in tqdm(lines, desc="Verifying"):
        video_path, expected_frame_count, _ = parse_line(line)

        if video_path is None:
            continue # Skip malformed lines, though our previous fix should prevent this.

        if not os.path.isdir(video_path):
            path_not_found += 1
            if path_not_found <= 5: # Print details for the first 5 not found
                print(f"\n  [Path Not Found] Path does not exist: {video_path}")
            continue

        actual_frames = glob.glob(os.path.join(video_path, '*.jpg'))
        actual_frames.extend(glob.glob(os.path.join(video_path, '*.png')))
        actual_frame_count = len(actual_frames)

        if actual_frame_count == expected_frame_count:
            valid_lines += 1
        else:
            mismatch_count += 1
            if mismatch_count <= 5: # Print details for the first 5 mismatches
                print(f"\n  [Mismatch Found] Path: {video_path}")
                print(f"    - Expected Frames (from file): {expected_frame_count}")
                print(f"    - Actual Frames (on disk):   {actual_frame_count}")

    print("\n--- Verification Summary ---")
    print(f"Total Lines Checked: {total_lines}")
    print(f"Valid Lines: {valid_lines}")
    print(f"Lines with Mismatched Frame Counts: {mismatch_count}")
    print(f"Lines with Paths Not Found: {path_not_found}")
    print("--------------------------\n")


if __name__ == '__main__':
    # We will verify all three annotation files
    BASE_PATH = '/content/drive/MyDrive/khoaluan/Dataset/DAiSEE'
    
    verify_annotations(os.path.join(BASE_PATH, 'daisee_train.txt'))
    verify_annotations(os.path.join(BASE_PATH, 'daisee_val.txt'))
    verify_annotations(os.path.join(BASE_PATH, 'daisee_test.txt'))
