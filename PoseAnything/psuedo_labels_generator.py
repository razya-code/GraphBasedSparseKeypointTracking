import argparse
import copy
import os
import random
import json  # ✅ FIXED: added missing import
import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms
from models import *
import torchvision.transforms.functional as F

from tools.visualization import plot_results

COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]
]

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
    parser = argparse.ArgumentParser(description='Pose Anything Demo')
    parser.add_argument('--support', help='Support image file')
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
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    os.makedirs(args.outdir, exist_ok=True)

    support_img = cv2.imread(args.support)
    if support_img is None:
        raise ValueError('Failed to read support image')

    query_imgs = [cv2.imread(qp) for qp in args.queries]
    if any(q is None for q in query_imgs):
        raise ValueError('One or more query images could not be read')

    preprocess_for_display = transforms.Compose([
        transforms.ToTensor(),
        Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])

    preprocessed_support_disp = preprocess_for_display(support_img).cpu().numpy().transpose(1, 2, 0) * 255
    frame = copy.deepcopy(preprocessed_support_disp.astype(np.uint8).copy())

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
            cv2.imshow("Source", frame)
        if event == cv2.EVENT_RBUTTONDOWN:
            kp_src = []
            frame = copy.deepcopy(support_img)
            cv2.imshow("Source", frame)

    def draw_line(event, x, y, flags, param):
        nonlocal skeleton, kp_src, frame, count, prev_pt, prev_pt_idx, marked_frame, color_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_point = min(kp_src, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            closest_point_index = kp_src.index(closest_point)
            c = COLORS[color_idx] if color_idx < len(COLORS) else random.choices(range(256), k=3)
            cv2.circle(frame, closest_point, 2, c, 1)
            if count == 0:
                prev_pt = closest_point
                prev_pt_idx = closest_point_index
                count += 1
                cv2.imshow("Source", frame)
            else:
                cv2.line(frame, prev_pt, closest_point, c, 2)
                cv2.imshow("Source", frame)
                count = 0
                skeleton.append((prev_pt_idx, closest_point_index))
                color_idx += 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            frame = copy.deepcopy(marked_frame)
            cv2.imshow("Source", frame)
            count = 0
            color_idx = 0
            skeleton = []
            prev_pt = None

    cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Source', 800, 600)
    cv2.setMouseCallback("Source", selectKP)
    cv2.imshow("Source", frame)
    print('Press any key when finished marking keypoints')
    while True:
        if cv2.waitKey(1) > 0:
            break

    marked_frame = copy.deepcopy(frame)
    cv2.setMouseCallback("Source", draw_line)
    print('Press any key when finished drawing skeleton')
    while True:
        if cv2.waitKey(1) > 0:
            break

    cv2.destroyAllWindows()

    kp_src = torch.tensor(kp_src).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])

    support_img_tensor = preprocess(support_img).flip(0)[None]
    query_img_tensors = [preprocess(q).flip(0)[None] for q in query_imgs]

    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([cfg.model.encoder_config.img_size] * 2)
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False

    kp_src_3d = torch.cat((kp_src, torch.zeros(kp_src.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.cat((torch.ones_like(kp_src), torch.zeros(kp_src.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(data_cfg, kp_src_3d, kp_src_3d_weight, sigma=1)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.tensor(target_weight_s).float()[None]

    model = build_posenet(cfg.model)
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model.eval()

    all_outputs = []  # query results

    for t, query_img_tensor in enumerate(query_img_tensors):
        data = {
            'img_s': [support_img_tensor],
            'img_q': query_img_tensor,
            'target_s': [target_s],
            'target_weight_s': [target_weight_s],
            'return_loss': False,
            'img_metas': [{
                'sample_skeleton': [skeleton],
                'query_skeleton': skeleton,
                'sample_joints_3d': [kp_src_3d],
                'query_joints_3d': kp_src_3d,
                'sample_center': [kp_src.mean(dim=0)],
                'query_center': kp_src.mean(dim=0),
                'sample_scale': [kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0]],
                'query_scale': kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0],
                'sample_rotation': [0],
                'query_rotation': 0,
                'sample_bbox_score': [1],
                'query_bbox_score': 1,
                'query_image_file': args.queries[t],
                'sample_image_file': [args.support],
            }]
        }

        with torch.no_grad():
            outputs = model(**data)

        keypoints = torch.tensor(outputs['points']).squeeze(0).cpu().numpy()
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if keypoints.ndim == 2 and keypoints.shape[1] == 3:
            keypoints = keypoints[:, :2]
        if keypoints.shape[1] != 2:
            raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")

        txt_path = os.path.join(args.outdir, f"frame_{t:04d}_keypoints.txt")
        with open(txt_path, 'w') as f:
            for idx in range(keypoints.shape[0]):
                x, y = keypoints[idx]
                f.write(f"{x:.2f},{y:.2f},{t}\n")

        adj_matrix = np.zeros((len(kp_src), len(kp_src)), dtype=int)
        for i, j in skeleton:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        np.savetxt(os.path.join(args.outdir, f"frame_{t:04d}_adj_matrix.txt"), adj_matrix, fmt='%d')

        plot_results(
            support_img_tensor[0].cpu().numpy().transpose(1, 2, 0),
            query_img_tensor[0].cpu().numpy().transpose(1, 2, 0),
            kp_src_3d,
            target_weight_s[0],
            None,
            target_weight_s[0],
            skeleton,
            None,
            torch.tensor(outputs['points']).squeeze(0),
            out_dir=args.outdir,
        )

        all_outputs.append((keypoints.tolist(), adj_matrix.tolist()))

    # Support output
    support_keypoints = kp_src.tolist()
    support_adj_matrix = np.zeros((len(kp_src), len(kp_src)), dtype=int)
    for i, j in skeleton:
        support_adj_matrix[i, j] = 1
        support_adj_matrix[j, i] = 1

    output_dict = {
        "support": {
            "keypoints": support_keypoints,
            "adjacency_matrix": support_adj_matrix.tolist()
        },
        "queries": [
            {
                "frame": i,
                "keypoints": keypoints,
                "adjacency_matrix": adj
            }
            for i, (keypoints, adj) in enumerate(all_outputs)
        ]
    }

    json_path = os.path.join(args.outdir, "pose_results.json")
    with open(json_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"\nSaved structured results to {json_path}")

if __name__ == '__main__':
    main()
