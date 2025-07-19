import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import uuid

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]


def plot_results(support_img, query_img, support_kp, support_w, query_kp, query_w, skeleton,
                 initial_proposals, prediction, radius=6, out_dir='./heatmaps'):
    img_names = [img.split("_")[0] for img in os.listdir(out_dir) if str_is_int(img.split("_")[0])]
    if len(img_names) > 0:
        name_idx = max([int(img_name) for img_name in img_names]) + 1
    else:
        name_idx = 0

    h, w, c = support_img.shape
    prediction = prediction[-1].cpu().numpy() * h
    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))

    for id, (img, w, keypoint) in enumerate(zip([support_img, query_img],
                                                [support_w, query_w],
                                                [support_kp, prediction])):
        f, axes = plt.subplots()
        plt.imshow(img)
        for k in range(keypoint.shape[0]):
            if w[k] > 0:
                kp = keypoint[k, :2]
                c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
                patch = plt.Circle(kp, radius, color=c)
                axes.add_patch(patch)
                axes.text(kp[0], kp[1], k)
                plt.draw()
        for l, limb in enumerate(skeleton):
            kp = keypoint[:, :2]
            if l > len(colors) - 1:
                c = [x / 255 for x in random.sample(range(0, 255), 3)]
            else:
                c = [x / 255 for x in colors[l]]
            if w[limb[0]] > 0 and w[limb[1]] > 0:
                patch = plt.Line2D([kp[limb[0], 0], kp[limb[1], 0]],
                                   [kp[limb[0], 1], kp[limb[1], 1]],
                                   linewidth=6, color=c, alpha=0.6)
                axes.add_artist(patch)
        plt.axis('off')  # command for hiding the axis.
        name = 'support' if id == 0 else 'query'
        plt.savefig(f'./{out_dir}/{str(name_idx)}_{str(name)}.png', bbox_inches='tight', pad_inches=0)
        if id == 1:
            plt.show()
        plt.clf()
        plt.close('all')


def str_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_embedding_visualization(embedding_tensor, save_path):
    """
    Save the embedding map as a PNG visualization.
    Args:
        embedding_tensor (torch.Tensor): shape (C, H, W)
        save_path (str): output PNG path
    """
    if not isinstance(embedding_tensor, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    
    if embedding_tensor.dim() == 3:
        emb = embedding_tensor
    elif embedding_tensor.dim() == 4:
        emb = embedding_tensor.squeeze(0)
    else:
        raise ValueError("Unsupported embedding tensor shape")

    # Compute mean over channels if necessary
    emb_np = emb.detach().cpu().numpy()
    if emb_np.shape[0] > 3:
        emb_np = np.mean(emb_np, axis=0)  # shape: H x W
        plt.imshow(emb_np, cmap='viridis')
    else:
        emb_np = np.transpose(emb_np, (1, 2, 0))  # shape: H x W x C
        plt.imshow(emb_np)

    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def visualize_feature_map(embedding_map: torch.Tensor, keypoint_coords: list, save_dir: str, prefix: str):
    """
    Visualize and save per-keypoint embedding heatmaps from a feature map.

    Args:
        embedding_map (torch.Tensor): shape (C, H, W) — output of the model.
        keypoint_coords (list of tuples): list of (x, y) coordinates in resized image space.
        save_dir (str): directory to save images.
        prefix (str): prefix for output filenames.
    """
    os.makedirs(save_dir, exist_ok=True)

    C, H, W = embedding_map.shape
    for idx, (x, y) in enumerate(keypoint_coords):
        if not (0 <= x < W and 0 <= y < H):
            print(f"[Warning] Keypoint {idx} out of bounds: ({x}, {y})")
            continue

        # Use bilinear interpolation to get embedding vector at subpixel (x, y)
        x0, y0 = int(x), int(y)
        embedding = embedding_map[:, y0, x0]  # (C,)

        # Compute similarity map: dot product between embedding and all locations
        dot_map = torch.einsum('c,chw->hw', embedding, embedding_map)  # (H, W)
        dot_map = dot_map.cpu().numpy()

        # Normalize for visualization
        dot_map -= dot_map.min()
        dot_map /= dot_map.max() + 1e-5

        plt.imshow(dot_map, cmap='jet')
        plt.title(f'Keypoint {idx} Feature Similarity')
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{prefix}_keypoint{idx:02d}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
