import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]


def ensure_numpy_keypoints(keypoint):
    if isinstance(keypoint, torch.Tensor):
        keypoint = keypoint.detach().cpu().numpy()
    keypoint = np.array(keypoint)
    if keypoint.ndim == 1:
        keypoint = keypoint.reshape(-1, 2)
    return keypoint


def resize_if_too_large(img, max_width=4096, max_height=2160):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (max_width, int(h * scale)))
    if h > max_height:
        scale = max_height / h
        return cv2.resize(img, (int(w * scale), max_height))
    return img


def plot_results(support_img, query_img, support_kp, support_w, query_kp, query_w, skeleton,
                 initial_proposals, prediction, radius=6, out_dir='./heatmaps'):
    os.makedirs(out_dir, exist_ok=True)

    img_names = [img.split("_")[0] for img in os.listdir(out_dir) if str_is_int(img.split("_")[0])]
    name_idx = max([int(img_name) for img_name in img_names]) + 1 if img_names else 0

    support_img = resize_if_too_large(support_img)
    query_img = resize_if_too_large(query_img)

    h, w, c = support_img.shape
    prediction = prediction[-1].detach().cpu().numpy() * h

    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))

    for id, (img, w_arr, keypoint) in enumerate(zip([support_img, query_img], [support_w, query_w], [support_kp, query_kp])):
        fig, axes = plt.subplots()
        plt.imshow(img)

        # Normalize keypoints
        if isinstance(keypoint, torch.Tensor):
            keypoint = keypoint.detach().cpu().numpy()
        elif isinstance(keypoint, list):
            keypoint = np.array(keypoint)
        if keypoint.ndim == 1:
            keypoint = keypoint.reshape(-1, 2)

        # Fallback to prediction if query_kp is empty or invalid
        if id == 1 and (keypoint.size == 0 or keypoint.shape[0] < 1):
            print(f"[Fallback] Using prediction for query keypoints.")
            keypoint = prediction
            w_arr = np.ones(keypoint.shape[0])

        for k in range(min(len(w_arr), keypoint.shape[0])):
            if w_arr[k] > 0:
                kp = keypoint[k, :2]
                c = (1, 0, 0, 0.75) if w_arr[k] == 1 else (0, 0, 1, 0.6)
                patch = plt.Circle(kp, radius, color=c)
                axes.add_patch(patch)
                axes.text(kp[0], kp[1], str(k), fontsize=8)

        if keypoint.shape[0] >= 2:
            for l, limb in enumerate(skeleton):
                kp = keypoint[:, :2]
                if max(limb) >= kp.shape[0] or max(limb) >= len(w_arr):
                    continue
                if w_arr[limb[0]] > 0 and w_arr[limb[1]] > 0:
                    c = [x / 255 for x in colors[l % len(colors)]]
                    patch = plt.Line2D([kp[limb[0], 0], kp[limb[1], 0]],
                                       [kp[limb[0], 1], kp[limb[1], 1]],
                                       linewidth=3, color=c, alpha=0.6)
                    axes.add_artist(patch)

        plt.axis('off')
        name = 'support' if id == 0 else 'query'
        output_path = os.path.join(out_dir, f"{name_idx}_{name}.png")

        try:
            fig.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        except Exception as e:
            print(f"[Error] Failed to save image {output_path}: {e}")

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
