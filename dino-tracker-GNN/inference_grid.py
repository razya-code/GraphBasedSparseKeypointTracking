import json
import os
import numpy as np
import torch
import argparse
from dino_tracker import DINOTracker
from models.model_inference import ModelInference

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(args):
    dino_tracker = DINOTracker(args)
    dino_tracker.load_fg_masks()
    model = dino_tracker.get_model()
    if args.iter is not None:
        model.load_weights(args.iter)

    trajectories_dir = dino_tracker.grid_trajectories_dir
    occlusions_dir = dino_tracker.grid_occlusions_dir
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(occlusions_dir, exist_ok=True)

    model_inference = ModelInference(
        model=model,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=dino_tracker.config['anchor_cosine_similarity_threshold'],
        cosine_similarity_threshold=dino_tracker.config['cosine_similarity_threshold'],
    )

    # Load keypoints JSON
    with open(args.keypoint_json_path, "r") as f:
        pose_data = json.load(f)

    queries = pose_data["queries"]
    query_entry = next((entry for entry in queries if entry["frame"] == args.start_frame), None)
    if query_entry is None:
        raise ValueError(f"No query keypoints found for frame {args.start_frame} in JSON.")

    # Normalized keypoints
    keypoints_np = np.array(query_entry["keypoints"], dtype=np.float32)

    # Unnormalize to original resolution
    orig_w, orig_h = dino_tracker.config["video_resw"], dino_tracker.config["video_resh"]
    keypoints_np[:, 0] *= orig_w
    keypoints_np[:, 1] *= orig_h

    # Convert to torch and scale to model resolution
    model_h, model_w = model.video.shape[-2], model.video.shape[-1]
    keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32, device=device)
    keypoints_tensor[:, 0] *= model_w / orig_w
    keypoints_tensor[:, 1] *= model_h / orig_h

    # Add frame index column
    frame_index_column = torch.full((keypoints_tensor.shape[0], 1), fill_value=args.start_frame, dtype=torch.float32, device=device)
    keypoints_tensor = torch.cat([keypoints_tensor, frame_index_column], dim=1)  # [N, 3]

    # === Run inference (pass plain tensor, not dict) ===
    trajectories, occlusions = model_inference.infer(keypoints_tensor, batch_size=args.batch_size)

    # Save outputs
    np.save(os.path.join(trajectories_dir, "keypoint_trajectories.npy"), trajectories[..., :2].cpu().numpy())
    np.save(os.path.join(occlusions_dir, "keypoint_occlusions.npy"), occlusions.cpu().numpy())

    print(f"✅ Saved trajectories to {trajectories_dir}")
    print(f"✅ Saved occlusions to {occlusions_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--iter", type=int, default=None, help="Iteration number of the model to load. If None, load latest.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--use-segm-mask", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--keypoint-json-path", type=str, required=True, help="Path to pose_results.json")
    args = parser.parse_args()
    run(args)
