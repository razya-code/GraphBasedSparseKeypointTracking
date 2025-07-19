import torch
from torch import nn
from timm import create_model
from .pam import PoseAnythingModel


class PoseAnythingSwinModel(PoseAnythingModel):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        encoder_config = model_cfg.encoder_config
        keypoint_head = model_cfg.get("keypoint_head", None)

        # ✅ Patch init_backbone BEFORE parent constructor
        def dummy_init_backbone(pretrained, encoder_config):
            return None, 'swin'

        self.init_backbone = dummy_init_backbone

        # ✅ FIX: Prevent crash if test_cfg is missing
        if not hasattr(model_cfg, 'test_cfg') or model_cfg.test_cfg is None:
            model_cfg.test_cfg = {}

        # ✅ Call parent constructor with fixed config
        super(PoseAnythingSwinModel, self).__init__(
            keypoint_head=keypoint_head,
            encoder_config=encoder_config,
            test_cfg=model_cfg.test_cfg
        )

        # ✅ Build Swin backbones with features_only=True
        swin_name = getattr(model_cfg, "backbone_name", "swin_tiny_patch4_window7_224")

        self.encoder_sample = create_model(
            swin_name, pretrained=False, img_size=encoder_config.img_size, features_only=True
        )
        if hasattr(self.encoder_sample, 'reset_classifier'):
            self.encoder_sample.reset_classifier(0)

        self.encoder_query = create_model(
            swin_name, pretrained=False, img_size=encoder_config.img_size, features_only=True
        )
        if hasattr(self.encoder_query, 'reset_classifier'):
            self.encoder_query.reset_classifier(0)

        self.backbone_type = 'swin'
        self.backbone = None

    def extract_features(self, img_s, img_q):
        feat_q = self.encoder_query(img_q)[-1]  # [B, C, H, W]
        feat_s = [self.encoder_sample(s_i)[-1] for s_i in img_s]  # List[B, C, H, W]
        feat_s = torch.stack(feat_s, dim=1)  # [B, N_shots, C, H, W]
        return feat_q, feat_s

    def forward(self, img_s, img_q, **kwargs):
        feat_q, feat_s = self.extract_features(img_s, img_q)
        return {
            "points": self.predict_from_features(feat_q)
        }

    def predict_from_features(self, feat_q):
        B, C, H, W = feat_q.shape
        heatmaps = feat_q.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        coords = self.heatmap_to_coord(heatmaps)
        return coords

    def heatmap_to_coord(self, heatmaps):
        B, K, H, W = heatmaps.shape
        coords = []
        for i in range(B):
            heatmap = heatmaps[i, 0]
            max_val = heatmap.max()
            y, x = (heatmap == max_val).nonzero(as_tuple=True)
            coords.append(torch.stack([x[0], y[0], torch.tensor(1.0)]))
        return torch.stack(coords).unsqueeze(0)
