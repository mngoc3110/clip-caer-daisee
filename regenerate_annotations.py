import os
import glob
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_PATH = '/content/drive/MyDrive/khoaluan/Dataset/DAiSEE'
DATASET_DIR = os.path.join(BASE_PATH, 'DataSet')
LABELS_DIR = os.path.join(BASE_PATH, 'Labels')
OUTPUT_DIR = BASE_PATH

SETS = {
    'train': {'csv': 'TrainLabels.csv', 'dir': 'Train'},
    'val': {'csv': 'ValidationLabels.csv', 'dir': 'Validation'},
    'test': {'csv': 'TestLabels.csv', 'dir': 'Test'}
}

# --- SCRIPT LOGIC ---

def regenerate_annotations():
    print("Starting annotation regeneration process (v4)...")

    for set_name, set_info in SETS.items():
        csv_path = os.path.join(LABELS_DIR, set_info['csv'])
        output_txt_filename = f"daisee_{set_name}.txt"
        output_txt_path = os.path.join(OUTPUT_DIR, output_txt_filename)

        print(f"\nProcessing {csv_path}...")

        if not os.path.isfile(csv_path):
            print(f"  - WARNING: CSV file not found, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"  - Found {len(df)} records. Generating {output_txt_filename}...")

        with open(output_txt_path, 'w') as txt_file:
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"  - {set_name}"):
                try:
                    clip_id = str(row['ClipID'])
                    label = int(row['Engagement'])
                    
                    # UPDATED v4: Use simplified, correct path structure
                    video_dir_path = os.path.join(DATASET_DIR, set_info['dir'], clip_id)

                    if os.path.isdir(video_dir_path):
                        frames = glob.glob(os.path.join(video_dir_path, '*.*')) # Use *.* to be safe
                        frame_count = len(frames)

                        if frame_count > 0:
                            txt_file.write(f"{video_dir_path} {frame_count} {label}\n")

                except KeyError as ke:
                    print(f"\n  - FATAL ERROR: Column not found in CSV: {ke}.")
                    return

        print(f"  - Successfully created {output_txt_path}")

    print("\nAnnotation regeneration complete!")

if __name__ == '__main__':
    regenerate_annotations()