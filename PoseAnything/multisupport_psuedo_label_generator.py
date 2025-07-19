# demo_fewshot.py
# Updated PoseAnything demo.py to support multiple support images (few-shot mode)

import argparse
import copy
import os
import random
import json
import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms
import torchvision.transforms.functional as F
from models import *
from tools.visualization import plot_results

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
          [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
          [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
          [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]

class Resize_Pad:
    def __init__(self, w=256, h=256):
        self.w = w
        self.h = h

    def __call__(self, image):
        _, w_1, h_1 = image.shape
        ratio_1 = w_1 / h_1
        if round(ratio_1, 2) != 1:
            if ratio_1 > 1:
                hp = int(w_1 - h_1) // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])
            else:
                wp = int(h_1 - w_1) // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
                return F.resize(image, [self.h, self.w])
        else:
            return F.resize(image, [self.h, self.w])

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Anything Few-Shot Demo')
    parser.add_argument('--support', nargs='+', help='List of support image files (few-shot)')
    parser.add_argument('--query', nargs='+', help='List of query image files (alias for --queries)')
    parser.add_argument('--queries', nargs='+', help='List of query image files')
    parser.add_argument('--config', default=None, help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--outdir', default='output', help='output directory')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    args = parser.parse_args()
    if args.queries is None and args.query is not None:
        args.queries = args.query
    return args

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    os.makedirs(args.outdir, exist_ok=True)

    support_imgs = [cv2.imread(sp) for sp in args.support]
    if any(s is None for s in support_imgs):
        raise ValueError('One or more support images could not be read')

    query_imgs = [cv2.imread(qp) for qp in args.queries]
    if any(q is None for q in query_imgs):
        raise ValueError('One or more query images could not be read')

    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([cfg.model.encoder_config.img_size] * 2)
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False

    support_kps = []
    support_skeleton = None
    support_img_tensors = []
    target_s_list = []
    target_weight_s_list = []

    for idx, support_img in enumerate(support_imgs):
        preprocess_for_display = transforms.Compose([
            transforms.ToTensor(),
            Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])
        preprocessed_disp = preprocess_for_display(support_img).cpu().numpy().transpose(1, 2, 0) * 255
        frame = copy.deepcopy(preprocessed_disp.astype(np.uint8).copy())

        kp_src = []
        skeleton = []
        count = 0
        prev_pt = None
        prev_pt_idx = None
        color_idx = 0

        def selectKP(event, x, y, flags, param):
            nonlocal kp_src, frame
            if event == cv2.EVENT_LBUTTONDOWN:
                kp_src.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)
                cv2.imshow("Support", frame)

        def draw_line(event, x, y, flags, param):
            nonlocal skeleton, kp_src, frame, count, prev_pt, prev_pt_idx, marked_frame, color_idx
            if event == cv2.EVENT_LBUTTONDOWN:
                closest_point = min(kp_src, key=lambda p: (p[0] - x)**2 + (p[1] - y)**2)
                closest_point_index = kp_src.index(closest_point)
                c = COLORS[color_idx] if color_idx < len(COLORS) else random.choices(range(256), k=3)
                cv2.circle(frame, closest_point, 2, c, 1)
                if count == 0:
                    prev_pt = closest_point
                    prev_pt_idx = closest_point_index
                    count += 1
                    cv2.imshow("Support", frame)
                else:
                    cv2.line(frame, prev_pt, closest_point, c, 2)
                    cv2.imshow("Support", frame)
                    count = 0
                    skeleton.append((prev_pt_idx, closest_point_index))
                    color_idx += 1

        cv2.namedWindow("Support", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Support", selectKP)
        cv2.imshow("Support", frame)
        print(f"Select keypoints for support image {idx+1}. Press any key when done.")
        while True:
            if cv2.waitKey(1) > 0:
                break

        marked_frame = copy.deepcopy(frame)
        cv2.setMouseCallback("Support", draw_line)
        print(f"Draw skeleton for support image {idx+1}. Press any key when done.")
        while True:
            if cv2.waitKey(1) > 0:
                break
        cv2.destroyAllWindows()

        kp_tensor = torch.tensor(kp_src).float()
        kp_3d = torch.cat((kp_tensor, torch.zeros(kp_tensor.shape[0], 1)), dim=-1)
        kp_3d_weight = torch.cat((torch.ones_like(kp_tensor), torch.zeros(kp_tensor.shape[0], 1)), dim=-1)
        target_s, target_weight_s = genHeatMap._msra_generate_target(data_cfg, kp_3d, kp_3d_weight, sigma=1)
        target_s_list.append(torch.tensor(target_s).float())
        target_weight_s_list.append(torch.tensor(target_weight_s).float())

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])
        support_tensor = preprocess(support_img).flip(0)
        support_img_tensors.append(support_tensor[None])

        support_kps = kp_tensor.tolist()
        support_skeleton = skeleton  # use skeleton from last support image

    target_s = torch.stack(target_s_list).mean(dim=0)[None]
    target_weight_s = torch.stack(target_weight_s_list).mean(dim=0)[None]

    model = build_posenet(cfg.model)
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model.eval()

    preprocess_q = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])
    query_img_tensors = [preprocess_q(q).flip(0)[None] for q in query_imgs]

    all_outputs = []
    for t, query_img_tensor in enumerate(query_img_tensors):
        data = {
            'img_s': support_img_tensors,
            'img_q': query_img_tensor,
            'target_s': [target_s],
            'target_weight_s': [target_weight_s],
            'return_loss': False,
            'img_metas': [{
                'sample_skeleton': [support_skeleton],
                'query_skeleton': support_skeleton,
                'sample_joints_3d': [torch.tensor(support_kps).float()],
                'query_joints_3d': torch.tensor(support_kps).float(),
                'sample_center': [torch.tensor(support_kps).float().mean(dim=0)],
                'query_center': torch.tensor(support_kps).float().mean(dim=0),
                'sample_scale': [torch.tensor(support_kps).float().max(dim=0)[0] - torch.tensor(support_kps).float().min(dim=0)[0]],
                'query_scale': torch.tensor(support_kps).float().max(dim=0)[0] - torch.tensor(support_kps).float().min(dim=0)[0],
                'sample_rotation': [0],
                'query_rotation': 0,
                'sample_bbox_score': [1],
                'query_bbox_score': 1,
                'query_image_file': args.queries[t],
                'sample_image_file': args.support
            }]
        }

        with torch.no_grad():
            outputs = model(**data)

        keypoints = torch.tensor(outputs['points']).squeeze(0).cpu().numpy()
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if keypoints.ndim == 2 and keypoints.shape[1] == 3:
            keypoints = keypoints[:, :2]

        txt_path = os.path.join(args.outdir, f"frame_{t:04d}_keypoints.txt")
        with open(txt_path, 'w') as f:
            for idx in range(keypoints.shape[0]):
                x, y = keypoints[idx]
                f.write(f"{x:.2f},{y:.2f},{t}\n")

        adj_matrix = np.zeros((len(support_kps), len(support_kps)), dtype=int)
        for i, j in support_skeleton:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        np.savetxt(os.path.join(args.outdir, f"frame_{t:04d}_adj_matrix.txt"), adj_matrix, fmt='%d')

        plot_results(
            support_img_tensors[0][0].cpu().numpy().transpose(1, 2, 0),
            query_img_tensor[0].cpu().numpy().transpose(1, 2, 0),
            torch.tensor(support_kps),
            target_weight_s[0],
            None,
            target_weight_s[0],
            support_skeleton,
            None,
            torch.tensor(outputs['points']).squeeze(0),
            out_dir=args.outdir,
        )

        all_outputs.append((keypoints.tolist(), adj_matrix.tolist()))

    # Save structured JSON
    output_dict = {
        "support": {
            "keypoints": support_kps,
            "adjacency_matrix": adj_matrix.tolist()
        },
        "queries": [
            {"frame": i, "keypoints": k, "adjacency_matrix": a}
            for i, (k, a) in enumerate(all_outputs)
        ]
    }
    with open(os.path.join(args.outdir, "pose_results.json"), "w") as f:
        json.dump(output_dict, f, indent=2)

    print("\nSaved structured results to pose_results.json")

if __name__ == '__main__':
    main()
