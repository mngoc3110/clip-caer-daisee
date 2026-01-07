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
    print("Starting annotation regeneration process (v6 - SUPER DEBUG)...")

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
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"  - {set_name}"):
                try:
                    clip_id = str(row['ClipID'])
                    label = int(row['Engagement'])
                    
                    intermediate_dir = clip_id[:6]
                    video_dir_path = os.path.join(DATASET_DIR, set_info['dir'], intermediate_dir, clip_id)

                    if not os.path.isdir(video_dir_path):
                        if idx < 5: # Only print debug info for the first 5 rows to avoid spam
                            print(f"\n[DEBUG] For ClipID '{clip_id}', check FAILED at row {idx}.")
                            print(f"  - Path being checked: {video_dir_path}")
                            print(f"  - os.path.isdir() returned: {os.path.isdir(video_dir_path)}")
                            print(f"  - os.path.exists() returned: {os.path.exists(video_dir_path)}")
                            
                            parent_dir = os.path.dirname(video_dir_path)
                            if os.path.exists(parent_dir):
                                print(f"  - Parent dir '{parent_dir}' EXISTS.")
                                try:
                                    contents = os.listdir(parent_dir)
                                    print(f"  - Contents of parent (first 10): {contents[:10]}")
                                    if clip_id in contents:
                                        print(f"  - STRANGE: ClipID '{clip_id}' IS in the parent's contents list!")
                                    else:
                                        print(f"  - As expected, ClipID '{clip_id}' is NOT in the parent's contents list.")
                                except Exception as e:
                                    print(f"  - Could not list contents of parent dir. Permissions issue? Error: {e}")
                            else:
                                print(f"  - Parent dir '{parent_dir}' does NOT exist either!")
                        continue
                    
                    frames = glob.glob(os.path.join(video_dir_path, '*.*'))
                    frame_count = len(frames)

                    if frame_count > 0:
                        txt_file.write(f"{video_dir_path} {frame_count} {label}\n")

                except KeyError as ke:
                    print(f"\n  - FATAL ERROR: Column not found in CSV: {ke}.")
                    return

        print(f"  - Successfully finished processing for {output_txt_path}")

    print("\nAnnotation regeneration complete!")

if __name__ == '__main__':
    regenerate_annotations()