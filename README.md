# GraphBasedSparseKeypointTracking

This project builds on two excellent repositories:
- [PoseAnything](https://github.com/orhir/PoseAnything)
- [DINO-Tracker](https://github.com/AssafSinger94/dino-tracker)

It adds a GNN-based refinement stage for sparse keypoint tracking across video frames.

---

## Setup

1. Clone this repository:

```bash
git clone git@github.com:razya-code/GraphBasedSparseKeypointTracking.git
cd GraphBasedSparseKeypointTracking
```

2. Install dependencies:

We recommend using the same conda environment setup as DINO-Tracker:

```bash
conda create -n dino-tracker python=3.9
conda activate dino-tracker
pip install -r requirements.txt
export PYTHONPATH=`pwd`:$PYTHONPATH
```

---

## Dependencies

- [DINO-Tracker](https://github.com/AssafSinger94/dino-tracker): For initial training and trajectory generation
- [PoseAnything](https://github.com/orhir/PoseAnything): For generating initial keypoints (pseudo-labels)

Please follow the setup and training instructions in both repositories before proceeding.

---

## Step-by-step Guide

### Step 1: Generate Pseudo Labels using PoseAnything

```bash
python PoseAnything/psuedo_labels_generator.py \
  --support path/to/support.jpg \
  --queries path/to/query1.jpg path/to/query2.jpg \
  --config configs/pose_model_config.py \
  --checkpoint checkpoints/model.pth \
  --outdir output
```

This will output:
```
PoseAnything/readyoutputs/<sequence_name>/pose_results.json
```

---

### Step 2: Adjust Pose Labels and Prepare for GNN

```bash
python scripts/run_pose_pipeline.py <sequence_name>
```

This generates:
- Adjusted JSON file with normalized keypoints and adjacency matrix
- Annotated frames with keypoints
- Output video showing tracked keypoints

**Note:**  
The generated video is based **only on PoseAnything** predictions and does not include results from the DINO or GNN tracker.

---

### Step 3: Preprocess and Train Original DINO Tracker

Run preprocessing as instructed in the original DINO-Tracker README:

```bash
python preprocessing/main_preprocessing.py \
  --config config/preprocessing.yaml \
  --data-path dataset/horsejump
```

Then train the original DINO Tracker:

```bash
python train.py \
  --config config/train.yaml \
  --data-path dataset/horsejump
```

---

### Step 4: Copy DINO Checkpoints to GNN Directory

After DINO training, copy the model files to the appropriate location in this repository:

```bash
cp -r path/to/dino-tracker/dataset/horsejump/models/ ./dataset/horsejump/
```

---

### Step 5: Train the GNN-Based Tracker

```bash
python train_gnn.py \
  --config config/train.yaml \
  --data-path dataset/horsejump
```

---

### Step 6: Inference with GNN-Enhanced Model

Run inference:

```bash
python inference_grid.py \
  --config config/train.yaml \
  --data-path dataset/horsejump
```

Then visualize:

```bash
python visualization/visualize_rainbow.py \
  --data-path dataset/horsejump \
  --plot-trails
```

The results (videos, images) will be saved under:
```
dataset/horsejump/visualizations/
```

---

## Folder Structure

```
GraphBasedSparseKeypointTracking/
├── PoseAnything/             # External repo used for pseudo-labels
├── dino-tracker-GNN/         # GNN training and inference
├── dataset/
│   └── horsejump/            # All training data and results
├── scripts/
│   └── run_pose_pipeline.py
├── output/                   # Visual output (videos)
└── README.md
```

---

## Citation

If you use this project, please cite the original works:

```
@misc{poseanything2023,
  author = {Or Hirsh, et al.},
  title = {PoseAnything: Flexible Keypoint Estimation for Arbitrary Skeletons},
  year = {2023}
}

@inproceedings{dino-tracker2024,
  author = {Tumanyan, Narek and Singer, Assaf and Bagon, Shai and Dekel, Tali},
  title = {DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video},
  booktitle = {ECCV},
  year = {2024}
}
```
