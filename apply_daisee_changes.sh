#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "--- Applying DAiSEE-specific changes ---"

echo "1. Creating dummy bounding box JSON files..."
# Create DAiSEE_data/face_bbox_dummy.json
cat <<EOF > DAiSEE_data/face_bbox_dummy.json
{}
EOF

# Create DAiSEE_data/body_bbox_dummy.json
cat <<EOF > DAiSEE_data/body_bbox_dummy.json
{}
EOF

echo "2. Updating models/Text.py with DAiSEE emotional mappings..."
# Content of models/Text.py
cat <<'EOF_TEXT_PY' > models/Text.py
class_names_5 = [
'Neutrality in learning state.',
'Enjoyment in learning state.',
'Confusion in learning state.',
'Fatigue in learning state.',
'Distraction.'
]

class_names_with_context_5 = [
'an expression of Neutrality in learning state.',
'an expression of Enjoyment in learning state.',
'an expression of Confusion in learning state.',
'an expression of Fatigue in learning state.',
'an expression of Distraction.'
]


##### onlyface
class_descriptor_5_only_face = [
'Relaxed mouth,open eyes,neutral eyebrows,smooth forehead,natural head position.',

'Upturned mouth,sparkling or slightly squinted eyes,raised eyebrows,relaxed forehead.',

'Furrowed eyebrows, slightly open mouth, squinting or narrowed eyes, tensed forehead.',

'Mouth opens in a yawn, eyelids droop, head tilts forward.',

'Averted gaze or looking away, restless or fidgety posture, shoulders shift restlessly.'
]

##### with_context
class_descriptor_5 = [
'Relaxed mouth,open eyes,neutral eyebrows,no noticeable emotional changes,engaged with study materials, or natural body posture.',

'Upturned mouth corners,sparkling eyes,relaxed eyebrows,focused on course content,or occasionally nodding in agreement.',

'Furrowed eyebrows, slightly open mouth, wandering or puzzled gaze, chin rests on the palm,or eyes lock on learning material.',

'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, or hand writing.',

'Shifting eyes, restless or fidgety posture, relaxed but unfocused expression,frequently checking phone,or averted gaze from study materials.'
]

# ================= DAiSEE EMOTIONAL MAPPING =================
# Mapping Engagement Levels (0-3) to Emotional States

class_names_daisee = [
    'Boredom and Distraction',      # Level 0
    'Fatigue and Passivity',        # Level 1
    'Calm Attention',               # Level 2
    'Strong Interest and Curiosity' # Level 3
]

class_names_with_context_daisee = [
    'a student feeling bored, distracted, or looking away from the screen.',
    'a student feeling tired, passive, sleepy, or zoning out.',
    'a student paying attention, looking calm and focused on the screen.',
    'a student showing strong interest, curiosity, and active engagement.'
]

class_descriptor_daisee = [
    'Face showing boredom, yawning, eyes looking away, head turning around, completely disengaged.',
    'Face showing fatigue, sleepy eyes, blank stare, resting head on hand, passive expression.',
    'Face showing calmness, serious expression, direct eye contact with screen, normal posture.',
    'Face showing excitement, raising eyebrows, leaning forward, intense focus, nodding or smiling.'
]
EOF_TEXT_PY

echo "3. Updating utils/builders.py to support DAiSEE dataset config..."
sed -i '' '/^    else:/i\
    elif args.dataset == "DAiSEE":\
        class_names = class_names_daisee\
        class_names_with_context = class_names_with_context_daisee\
        class_descriptor = class_descriptor_daisee' utils/builders.py

echo "4. Updating dataloader/video_dataloader.py to apply transforms for DAiSEE..."
sed -i '' 's/if dataset_name == "RAER":/if dataset_name == "RAER" or dataset_name == "DAiSEE":/' dataloader/video_dataloader.py

echo "5. Updating DAiSEE_data/make.py to generate full relative paths..."
cat <<'EOF_MAKE_PY' > DAiSEE_data/make.py
import csv
from pathlib import Path

ROOT_DATASET = Path("DataSet")   # Train/Validation/Test
ROOT_LABELS  = Path("Labels")
TARGET_DIM = "Engagement"        # Boredom | Engagement | Confusion | Frustration

# ưu tiên nếu có folder frames/; nếu không có thì lấy ảnh ngay trong video_dir
FRAMES_DIRNAME = "frames"

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")
IMG_EXTS = (".jpg", ".jpeg", ".png")


def map_level_to_label(level_0_to_3: int) -> int:
    # CSV 0..3 -> txt 1..4 (vì dataloader return label-1)
    if level_0_to_3 not in (0, 1, 2, 3):
        raise ValueError(f"Invalid level: {level_0_to_3} (expected 0..3)")
    return level_0_to_3 + 1


def load_labels_csv(csv_path: Path, target_dim: str) -> dict:
    """
    Return dict keyed by clip STEM (no extension): 
      "1100011002" -> level 0..3
    CSV ClipID thường là "1100011002.avi"
    """
    mapping = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "ClipID" not in cols:
            raise RuntimeError(f"{csv_path} missing ClipID. Found: {cols}")
        if target_dim not in cols:
            raise RuntimeError(f"{csv_path} missing {target_dim}. Found: {cols}")

        for row in reader:
            clipid = (row.get("ClipID") or "").strip()
            if not clipid:
                continue
            stem = Path(clipid).stem  # bỏ .avi
            val = (row.get(target_dim) or "").strip()
            if val == "":
                continue
            try:
                level = int(float(val))
            except ValueError:
                continue
            mapping[stem] = level
    return mapping


def find_video_file(video_dir: Path) -> Path | None:
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(video_dir.glob(f"*{ext}"))
        vids.extend(video_dir.glob(f"*{ext.upper()}"))
    if not vids:
        return None
    # chọn file đầu tiên (thường chỉ có 1)
    return sorted(vids)[0]


def count_frames_in_dir(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(dir_path.glob(f"*{ext}"))
        imgs.extend(dir_path.glob(f"*{ext.upper()}"))
    return len(imgs)


def get_frames_folder(video_dir: Path) -> Path | None:
    """
    Ưu tiên frames/ nếu tồn tại và có ảnh.
    Nếu không, xem ảnh nằm trực tiếp trong video_dir.
    """
    frames_dir = video_dir / FRAMES_DIRNAME
    n1 = count_frames_in_dir(frames_dir)
    if n1 > 0:
        return frames_dir

    n2 = count_frames_in_dir(video_dir)
    if n2 > 0:
        return video_dir

    return None


def iter_video_folders(split_dir: Path):
    for subject in sorted(split_dir.iterdir()):
        if not subject.is_dir() or subject.name.startswith("."):
            continue
        for video_dir in sorted(subject.iterdir()):
            if not video_dir.is_dir() or video_dir.name.startswith("."):
                continue
            yield subject, video_dir


def make_split_txt(split_name: str, labels_map: dict, out_path: Path) -> None:
    split_dir = ROOT_DATASET / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    lines = []
    total = 0
    missing_label = 0
    missing_video = 0
    missing_frames = 0

    for _, video_dir in iter_video_folders(split_dir):
        total += 1

        video_file = find_video_file(video_dir)
        if video_file is None:
            missing_video += 1
            continue

        stem = video_file.stem  # bỏ .mp4/.avi...
        if stem not in labels_map:
            missing_label += 1
            continue

        frames_folder = get_frames_folder(video_dir)
        if frames_folder is None:
            missing_frames += 1
            continue

        n_frames = count_frames_in_dir(frames_folder)
        label_txt = map_level_to_label(labels_map[stem])

        # record.path cần là path tương đối từ ROOT_DATASET
        rel_path = frames_folder.relative_to(ROOT_DATASET).as_posix()
        # Fix path to be relative to project root
        full_rel_path = f"DAiSEE_data/DataSet/{rel_path}"
        lines.append(f"{full_rel_path} {n_frames} {label_txt}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[{split_name}] wrote: {out_path}")
    print(f"  total video folders scanned: {total}")
    print(f"  kept (have label + frames): {len(lines)}")
    print(f"  missing video file: {missing_video}")
    print(f"  missing label: {missing_label}")
    print(f"  missing frames: {missing_frames}")


def main():
    csv_files = {
        "Train": ROOT_LABELS / "TrainLabels.csv",
        "Validation": ROOT_LABELS / "ValidationLabels.csv",
        "Test": ROOT_LABELS / "TestLabels.csv",
    }

    label_maps = {}
    for split, csv_path in csv_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing label CSV: {csv_path}")
        label_maps[split] = load_labels_csv(csv_path, TARGET_DIM)

    make_split_txt("Train",      label_maps["Train"],      Path("daisee_train.txt"))
    make_split_txt("Validation", label_maps["Validation"], Path("daisee_val.txt"))
    make_split_txt("Test",       label_maps["Test"],       Path("daisee_test.txt"))

    print("\nDone.")
    print("NOTE: txt labels are 1..4, dataloader returns label-1 -> final classes 0..3.")


if __name__ == "__main__":
    main()
EOF_MAKE_PY

echo "6. Running DAiSEE_data/make.py to regenerate annotation files..."
(cd DAiSEE_data && python make.py)

echo "7. Updating trainer.py to print confusion matrix during training/validation..."
cat <<'EOF_TRAINER_PY' > trainer.py
# trainer.py
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.utils import AverageMeter, ProgressMeter

class Trainer:
    """A class that encapsulates the training and validation logic."""
    def __init__(self, model, criterion, optimizer, scheduler, device,log_txt_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = 10
        self.log_txt_path = log_txt_path

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        """Runs one epoch of training or validation."""
        if is_train:
            self.model.train()
            prefix = f"Train Epoch: [{epoch_str}]"
        else:
            self.model.eval()
            prefix = f"Valid Epoch: [{epoch_str}]"

        losses = AverageMeter('Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        progress = ProgressMeter(
            len(loader), 
            [losses, war_meter],
            prefix=prefix, 
            log_txt_path=self.log_txt_path  
        )

        all_preds = []
        all_targets = []

        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)

                # Forward pass
                output = self.model(images_face, images_body)
                loss = self.criterion(output, target)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Record metrics
                preds = output.argmax(dim=1)
                correct_preds = preds.eq(target).sum().item()
                acc = (correct_preds / target.size(0)) * 100.0

                losses.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

                if i % self.print_freq == 0:
                    progress.display(i)
        
        # Calculate epoch-level metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        war = war_meter.avg # Weighted Average Recall (WAR) is just the overall accuracy
        
        # Unweighted Average Recall (UAR)
        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) # Add epsilon to avoid division by zero
        uar = np.nanmean(class_acc) * 100

        # Format Confusion Matrix to string for logging
        cm_str = "\nConfusion Matrix:\n"
        cm_str += np.array2string(cm, separator=', ')
        
        print(cm_str) # Print to console
        with open(self.log_txt_path, 'a') as f:
            f.write(cm_str + '\n')

        logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        with open(self.log_txt_path, 'a') as f:
            f.write('Current WAR: {war:.3f}'.format(war=war) + '\n')
            f.write('Current UAR: {uar:.3f}'.format(uar=uar) + '\n')
        return war, uar, losses.avg, cm
        
    def train_epoch(self, train_loader, epoch_num):
        """Executes one full training epoch."""
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)
    
    def validate(self, val_loader, epoch_num_str="Final"):
        """Executes one full validation run."""
        return self._run_one_epoch(val_loader, epoch_num_str, is_train=False)
EOF_TRAINER_PY

echo "8. Creating train_daisee.sh for easy execution..."
cat <<'EOF_TRAIN_DAISEE_SH' > train_daisee.sh
#!/bin/bash

# Script to train on DAiSEE dataset (Engagement only)
# Uses dummy bounding boxes (full frame) and custom text prompts.

python main.py \
    --exper-name daisee_engagement_finetune \
    --dataset DAiSEE \
    --gpu mps \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.001 \
    --lr-image-encoder 1e-5 \
    --lr-prompt-learner 0.001 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --milestones 10 15 \
    --gamma 0.1 \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --print-freq 10 \
    --root-dir DAiSEE_data/DataSet \
    --train-annotation ../daisee_train.txt \
    --test-annotation ../daisee_test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face DAiSEE_data/face_bbox_dummy.json \
    --bounding-box-body DAiSEE_data/body_bbox_dummy.json \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True
EOF_TRAIN_DAISEE_SH

echo "All DAiSEE-specific changes have been summarized in apply_daisee_changes.sh."
echo "You can review the script and run it using: bash apply_daisee_changes.sh"
echo "After applying changes, you can start training with: sh train_daisee.sh"
