# CLIP-CAER for Engagement Estimation on DAiSEE Dataset

This project adapts the **CLIP-CAER** (Context-Aware Emotion Recognition) architecture for the task of **student engagement level estimation** using the [DAiSEE (Dataset for Affective States in E-learning)](http://daisee-dataset.com/) dataset.

The model leverages the power of OpenAI's CLIP for robust visual feature extraction, combined with a temporal transformer to capture the dynamics of video sequences. It uses prompt learning to create adaptable text-based classifiers for different engagement levels.

## Features
- **CLIP-based Visual Encoding**: Utilizes the pretrained CLIP ViT-B/32 image encoder for powerful and generalizable spatial feature extraction.
- **Dual-Stream Input**: Processes both cropped faces and full-body frames to capture a holistic view of student behavior.
- **Temporal Modeling**: Employs a Temporal Transformer to model the relationships between video frames over time.
- **Prompt Learning**: Uses a learnable prompt mechanism (`CoOp`) to adapt text-based classifiers to the specific nuances of the engagement detection task, making it more flexible than traditional linear classifiers.
- **DAiSEE Dataset Integration**: Comes fully configured to train and evaluate on the DAiSEE dataset for engagement level classification (boredom, confusion, engagement, frustration).

## Project Structure
```
.
├── DAiSEE_data/            # Root directory for dataset
│   ├── DataSet/            # Processed video frames should be here
│   └── Labels/             # Annotation files (train.txt, test.txt, etc.)
├── dataloader/
│   ├── video_dataloader.py # Main dataloader script
│   └── video_transform.py  # Video data augmentation and transformations
├── models/
│   ├── Generate_Model.py   # Defines the main CLIP-CAER architecture
│   ├── Prompt_Learner.py   # Implements the context prompt learner
│   ├── Temporal_Model.py   # Implements the Temporal Transformer
│   └── clip/               # OpenAI's CLIP model source
├── outputs/                # Stores logs, checkpoints, and results
├── main.py                 # Main script for training and evaluation
├── trainer.py              # Contains the training and validation loops
├── train_daisee.sh         # Example script for training
└── valid.sh                # Example script for evaluation
```

## Setup & Installation

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd clip-caer-daisee
```

**2. Create a Python virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
This project relies on PyTorch, torchvision, and the original OpenAI CLIP package.
```bash
pip install torch torchvision
pip install ftfy regex tqdm numpy pandas scikit-learn matplotlib
pip install git+https://github.com/openai/CLIP.git
```

## Dataset Preparation (DAiSEE)

**1. Download the Data**:
Download the DAiSEE dataset from the official website: [http://daisee-dataset.com/](http://daisee-dataset.com/). You will need to fill out their agreement form to get access.

**2. Extract Video Frames**:
The code expects video frames to be pre-extracted into folders. You will need to process the downloaded videos (`.avi` files) into individual frames (`.jpg` or `.png`). Place the extracted frames inside `DAiSEE_data/DataSet/`. The structure should look like this:
```
DAiSEE_data/
└── DataSet/
    ├── 1100031002/
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   └── ...
    ├── 1100031003/
    │   ├── 00001.jpg
    │   └── ...
    └── ...
```

**3. Create Annotation Files**:
The dataloader requires text files that map video frames to their labels. Create `train.txt`, `test.txt`, and `validation.txt` inside `DAiSEE_data/Labels/`. Each line in these files should follow this format:
`<path_to_frame_folder> <number_of_frames> <label_index>`

- `label_index` for DAiSEE corresponds to:
  - `0`: Boredom
  - `1`: Engagement
  - `2`: Confusion
  - `3`: Frustration

**Example `train.txt` line:**
```
DAiSEE_data/DataSet/1100031002 150 1
DAiSEE_data/DataSet/2100042001 210 0
```

**4. (Optional) Bounding Box Files**:
The dataloader can use pre-computed bounding boxes for face and body detection to improve performance. If you have these, format them as JSON files (`face.json`, `body.json`) and place them in the `DAiSEE_data/` directory. The path to these files is specified in the training script. If not provided, the code will default to using the full frame.

## How to Run

Parameters such as learning rate, batch size, and GPU ID can be configured directly in the shell scripts or passed as command-line arguments.

### Training
Modify the `train_daisee.sh` script to point to your DAiSEE annotation files. Then, run the script to start training:

```bash
#!/bin/bash

# train_daisee.sh

python main.py \
    --mode train \
    --exper-name daisee_engagement_finetune \
    --dataset DAiSEE \
    --gpu 0 \
    --epochs 20 \
    --batch-size 4 \
    --lr 0.003 \
    --lr-image-encoder 1e-5 \
    --lr-prompt-learner 0.001 \
    --num-segments 16 \
    --root-dir ./ \
    --train-annotation DAiSEE_data/Labels/train.txt \
    --test-annotation DAiSEE_data/Labels/validation.txt \
    --clip-path ViT-B/32 \
    # --- Optional: Update if you have bounding box files ---
    # --bounding-box-face DAiSEE_data/face.json \
    # --bounding-box-body DAiSEE_data/body.json \
```
**Execute the script:**
```bash
bash train_daisee.sh
```

### Evaluation
To evaluate a trained model, modify `valid.sh` to specify the path to your best model checkpoint (`--eval-checkpoint`) and the test set annotation file.

```bash
#!/bin/bash

# valid.sh

python main.py \
    --mode eval \
    --exper-name daisee_evaluation \
    --dataset DAiSEE \
    --gpu 0 \
    --eval-checkpoint "outputs/daisee_engagement_finetune-[date]-[time]/model_best.pth" \
    --root-dir ./ \
    --test-annotation DAiSEE_data/Labels/test.txt \
    --clip-path ViT-B/32 \
    # --- Optional: Update if you have bounding box files ---
    # --bounding-box-face DAiSEE_data/face.json \
    # --bounding-box-body DAiSEE_data/body.json \
```
**Execute the script:**
```bash
bash valid.sh
```

## Outputs
All training artifacts are saved in the `outputs/` directory, organized by experiment name and timestamp. This includes:
- `log.txt`: A detailed log of the training process, including hyperparameters, epoch-level performance, and final confusion matrix.
- `log.png`: A plot showing the training and validation loss/accuracy curves over epochs.
- `model.pth`: The model checkpoint from the latest epoch.
- `model_best.pth`: The model checkpoint with the highest validation Unweighted Average Recall (UAR).
- `confusion_matrix.png`: A visualization of the final confusion matrix on the validation/test set.

## Acknowledgments
This codebase is an adaptation of the original [CLIP-CAER](https://github.com/weixiaoh/CLIP-CAER) project. We thank the authors for their significant contribution to the field of emotion recognition.