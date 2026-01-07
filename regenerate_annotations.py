import os
import glob
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# Base path for the DAiSEE dataset in your Google Drive
BASE_PATH = '/content/drive/MyDrive/khoaluan/Dataset/DAiSEE'
# Directory containing the video frame folders (e.g., '1100031002', etc.)
DATASET_DIR = os.path.join(BASE_PATH, 'DataSet')
# Directory where the original CSV label files are
LABELS_DIR = os.path.join(BASE_PATH, 'Labels')
# Directory where the new .txt annotation files will be saved
OUTPUT_DIR = BASE_PATH

# Define the sets to process
SETS = {
    'train': 'TrainLabels.csv',
    'val': 'ValidationLabels.csv',
    'test': 'TestLabels.csv'
}

# --- SCRIPT LOGIC ---

def regenerate_annotations():
    """
    Reads original DAiSEE CSV files, counts actual frames on disk,
    and generates new .txt annotation files in the required format.
    """
    print("Starting annotation regeneration process...")

    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Video frames directory not found at: {DATASET_DIR}")
        print("Please make sure the DATASET_DIR path is correct.")
        return

    for set_name, csv_filename in SETS.items():
        csv_path = os.path.join(LABELS_DIR, csv_filename)
        output_txt_filename = f"daisee_{set_name}.txt"
        output_txt_path = os.path.join(OUTPUT_DIR, output_txt_filename)

        print(f"\nProcessing {csv_path}...")

        if not os.path.isfile(csv_path):
            print(f"  - WARNING: CSV file not found, skipping: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  - ERROR: Could not read CSV file. Error: {e}")
            continue

        print(f"  - Found {len(df)} records. Generating {output_txt_filename}...")

        with open(output_txt_path, 'w') as txt_file:
            # Using tqdm for a progress bar
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"  - {set_name}"):
                try:
                    clip_id = str(row['ClipID'])
                    # We use the 'Engagement' column for the label
                    label = int(row['Engagement'])
                    
                    video_dir_path = os.path.join(DATASET_DIR, clip_id)

                    if os.path.isdir(video_dir_path):
                        # Count all .jpg and .png files
                        frames = glob.glob(os.path.join(video_dir_path, '*.jpg'))
                        frames.extend(glob.glob(os.path.join(video_dir_path, '*.png')))
                        frame_count = len(frames)

                        if frame_count > 0:
                            # Write the line in the format: <path_to_video_dir> <frame_count> <label>
                            txt_file.write(f"{video_dir_path} {frame_count} {label}\n")
                        else:
                            # This directory exists but is empty
                            # print(f"\n  - Warning: No frames found in {video_dir_path}, skipping.")
                            pass
                    else:
                        # The directory for this ClipID does not exist
                        # print(f"\n  - Warning: Directory not found for ClipID {clip_id}, skipping.")
                        pass

                except KeyError as ke:
                    print(f"\n  - FATAL ERROR: Column not found in CSV: {ke}. Please check the column names.")
                    print("  - Aborting process.")
                    return
                except Exception as e:
                    print(f"\n  - ERROR processing row {row}: {e}")

        print(f"  - Successfully created {output_txt_path}")

    print("\nAnnotation regeneration complete!")

if __name__ == '__main__':
    regenerate_annotations()
