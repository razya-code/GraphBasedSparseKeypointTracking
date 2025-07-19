import math
import torch
import json


class RangeNormalizer(torch.nn.Module):
    def __init__(self, shapes: tuple, device='cuda'):
        super().__init__()
        normalizer = torch.tensor(shapes).float().to(device) - 1
        self.register_buffer("normalizer", normalizer)

    def forward(self, x, dst=(0, 1), dims=[0, 1, 2]):
        normalized_x = x.clone()
        normalized_x[:, dims] = x[:, dims] / self.normalizer[dims]
        normalized_x[:, dims] = (dst[1] - dst[0]) * normalized_x[:, dims] + dst[0]
        return normalized_x

    def unnormalize(self, normalized_x: torch.tensor, src=(0, 1), dims=[0, 1, 2]):
        x = normalized_x.clone()
        x[:, dims] = (normalized_x[:, dims] - src[0]) / (src[1] - src[0])
        x[:, dims] = x[:, dims] * self.normalizer[dims]
        return x


class LongRangeSampler(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 fg_trajectories=None,
                 bg_trajectories=None,
                 fg_traj_ratio=0.5,
                 num_frames=None,
                 keep_in_cpu=False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.fg_traj_ratio = fg_traj_ratio
        self.keep_in_cpu = keep_in_cpu
        self.max_traj_size = 200_000
        self.gpu_batch_index = 0
        self.iter = 0

        if not self.keep_in_cpu:
            self.fg_valid_trajectories, self.fg_can_sample = self.get_valid_trajectories(fg_trajectories)
            self.bg_valid_trajectories, self.bg_can_sample = self.get_valid_trajectories(bg_trajectories)
            self.vid_len = self.fg_valid_trajectories.shape[1]
        else:
            self.fg_valid_trajectories_complete, self.fg_can_sample_complete = self.get_valid_trajectories(fg_trajectories)
            del fg_trajectories
            torch.cuda.empty_cache()
            self.fg_valid_trajectories = self.fg_valid_trajectories_complete[:self.max_traj_size].cuda()
            self.fg_can_sample = self.fg_can_sample_complete[:self.max_traj_size].cuda()
            self.n_batches_fg = math.ceil(self.fg_valid_trajectories_complete.shape[0] / self.max_traj_size)

            self.bg_valid_trajectories_complete, self.bg_can_sample_complete = self.get_valid_trajectories(bg_trajectories)
            del bg_trajectories
            torch.cuda.empty_cache()
            self.bg_valid_trajectories = self.bg_valid_trajectories_complete[:self.max_traj_size].cuda()
            self.bg_can_sample = self.bg_can_sample_complete[:self.max_traj_size].cuda()
            self.n_batches_bg = math.ceil(self.bg_valid_trajectories_complete.shape[0] / self.max_traj_size)

    def get_valid_trajectories(self, trajectories):
        can_sample = trajectories.isnan().any(dim=-1).logical_not()
        valid_trajs_idx = (can_sample.sum(dim=1) > 1)
        valid_trajectories = trajectories[valid_trajs_idx]
        can_sample = can_sample[valid_trajs_idx]
        return valid_trajectories, can_sample

    def load_next_batch(self):
        if not self.keep_in_cpu:
            return
        self.gpu_batch_index += 1
        if hasattr(self, "fg_valid_trajectories_complete"):
            del self.fg_can_sample, self.fg_valid_trajectories
            fg_batch_index = self.gpu_batch_index % self.n_batches_fg
            start_index, end_index = fg_batch_index * self.max_traj_size, min((fg_batch_index + 1) * self.max_traj_size, self.fg_valid_trajectories_complete.shape[0])
            self.fg_valid_trajectories = self.fg_valid_trajectories_complete[start_index:end_index].cuda()
            self.fg_can_sample = self.fg_can_sample_complete[start_index:end_index].cuda()
        if hasattr(self, "bg_valid_trajectories_complete"):
            del self.bg_can_sample, self.bg_valid_trajectories
            bg_batch_index = self.gpu_batch_index % self.n_batches_bg
            start_index, end_index = bg_batch_index * self.max_traj_size, min((bg_batch_index + 1) * self.max_traj_size, self.bg_valid_trajectories_complete.shape[0])
            self.bg_valid_trajectories = self.bg_valid_trajectories_complete[start_index:end_index].cuda()
            self.bg_can_sample = self.bg_can_sample_complete[start_index:end_index].cuda()

    def get_fg_batch_size(self):
        return int(self.batch_size * self.fg_traj_ratio)

    def forward(self):
        raise NotImplementedError


class DinoTrackerSampler(LongRangeSampler):
    def __init__(self,
                 batch_size,
                 range_normalizer,
                 dst_range,
                 fg_trajectories=None,
                 bg_trajectories=None,
                 fg_traj_ratio=0.5,
                 num_frames=None,
                 keep_in_cpu=False,
                 pseudo_labels_path=None,
                 video_resw=854,
                 video_resh=476) -> None:
        super().__init__(batch_size,
                         fg_trajectories=fg_trajectories,
                         bg_trajectories=bg_trajectories,
                         fg_traj_ratio=fg_traj_ratio,
                         num_frames=num_frames,
                         keep_in_cpu=keep_in_cpu)
        self.range_normalizer = range_normalizer
        self.dst_range = dst_range
        self.pseudo_labels_path = pseudo_labels_path
        self.video_resw = video_resw
        self.video_resh = video_resh

        if self.pseudo_labels_path:
            with open(self.pseudo_labels_path, "r") as f:
                self.pseudo_labels_data = json.load(f)
            self.pseudo_label_frames = set(query["frame"] for query in self.pseudo_labels_data["queries"])
        else:
            self.pseudo_labels_data = None
            self.pseudo_label_frames = set()

    # def get_pseudo_labels_for_frame(self, frame_id):
    #     if self.pseudo_labels_data is None:
    #         raise ValueError("Pseudo label JSON data not loaded. Please specify pseudo_labels_path.")
    #     for query in self.pseudo_labels_data["queries"]:
    #         if query["frame"] == int(frame_id):
    #             keypoints = torch.tensor(query["keypoints"], dtype=torch.float32)
    #             keypoints[:, 0] *= self.video_resw
    #             keypoints[:, 1] *= self.video_resh
    #             return keypoints
    #     raise ValueError(f"Frame {frame_id} not found in pseudo label file.")
    def get_pseudo_labels_for_frame(self, frame_id):
        if self.pseudo_labels_data is None:
            raise ValueError("Pseudo label JSON data not loaded. Please specify pseudo_labels_path.")
        for query in self.pseudo_labels_data["queries"]:
            if query["frame"] == int(frame_id):
                keypoints = torch.tensor(query["keypoints"], dtype=torch.float32)
                keypoints[:, 0] *= self.video_resw
                keypoints[:, 1] *= self.video_resh

                adj_matrix = torch.tensor(query["adjacency_matrix"], dtype=torch.float32)

                return keypoints, adj_matrix
        raise ValueError(f"Frame {frame_id} not found in pseudo label file.")

    def get_point_correspondences_for_num_frames(self, valid_trajectories, can_sample, batch_size):
        b, t, _ = valid_trajectories.shape
        done_selecting_frames = False
        while not done_selecting_frames:
            times = torch.tensor(sorted(self.pseudo_label_frames), device=valid_trajectories.device)
            valid_times = times[times < t]
            if valid_times.numel() < self.num_frames:
                raise ValueError("Not enough valid pseudo label frames to sample.")

            t_selector = torch.randperm(valid_times.shape[0], device=valid_trajectories.device)[:self.num_frames]
            frame_indices = valid_times[t_selector]

            can_sample_at_frame_indices = can_sample.float()[:, frame_indices].sum(dim=1) >= 2
            can_sample_current = can_sample[can_sample_at_frame_indices]
            if len(can_sample_current) >= 2:
                trajectories = valid_trajectories[can_sample_at_frame_indices]
                done_selecting_frames = True

        batch_size_selector = torch.randperm(trajectories.shape[0], device=valid_trajectories.device)[:batch_size]
        can_sample_current = can_sample_current[batch_size_selector]
        can_sample_in_frame_indices = can_sample_current[:, frame_indices]
        can_sample_current[:, :] = False
        can_sample_current[:, frame_indices] = can_sample_in_frame_indices
        only_2_ts = can_sample_current.float().multinomial(2, replacement=False)
        t1, t2 = only_2_ts.unbind(dim=1)
        t1_points = trajectories[batch_size_selector, t1]
        t2_points = trajectories[batch_size_selector, t2]
        t1_points = torch.cat([t1_points, t1.unsqueeze(dim=-1)], dim=-1)
        t2_points = torch.cat([t2_points, t2.unsqueeze(dim=-1)], dim=-1)

        return t1_points, t2_points

    # def forward(self):
    #     assert self.num_frames is not None, "num_frames must be specified"

    #     fg_batch_size = self.get_fg_batch_size()
    #     bg_batch_size = self.batch_size - fg_batch_size

    #     fg_t1_points, fg_t2_points = self.get_point_correspondences_for_num_frames(
    #         self.fg_valid_trajectories, self.fg_can_sample, fg_batch_size)
    #     bg_t1_points, bg_t2_points = self.get_point_correspondences_for_num_frames(
    #         self.bg_valid_trajectories, self.bg_can_sample, bg_batch_size)

    #     t1_points = torch.cat([fg_t1_points, bg_t1_points], dim=0)
    #     t2_points = torch.cat([fg_t2_points, bg_t2_points], dim=0)

    #     frames_set_t = torch.cat((t1_points[:, 2], t2_points[:, 2])).unique().int()
    #     source_frame_indices = torch.cat([(frames_set_t == i).nonzero() for i in t1_points[:, 2]])[:, 0]
    #     target_frame_indices = torch.cat([(frames_set_t == i).nonzero() for i in t2_points[:, 2]])[:, 0]

    #     t1_points_normalized = self.range_normalizer(t1_points, dst=self.dst_range)
    #     t2_points_normalized = self.range_normalizer(t2_points, dst=self.dst_range)
    #     t1_points[:, 2] = t1_points_normalized[:, 2]

    #     target_frame_idx = int(t2_points[0, 2].item())
    #     pseudo_labels = self.get_pseudo_labels_for_frame(target_frame_idx)

    #     sample = {
    #         "frames_set_t": frames_set_t,
    #         "source_frame_indices": source_frame_indices,
    #         "target_frame_indices": target_frame_indices,
    #         "t1_points_normalized": t1_points_normalized,
    #         "t2_points_normalized": t2_points_normalized,
    #         "t1_points": t1_points,
    #         "target_times": t2_points[:, 2],
    #         "pseudo_labels": pseudo_labels[:, :2],
    #     }

    #     return sample

    def forward(self):
        assert self.num_frames is not None, "num_frames must be specified"

        fg_batch_size = self.get_fg_batch_size()
        bg_batch_size = self.batch_size - fg_batch_size

        fg_t1_points, fg_t2_points = self.get_point_correspondences_for_num_frames(
            self.fg_valid_trajectories, self.fg_can_sample, fg_batch_size)
        bg_t1_points, bg_t2_points = self.get_point_correspondences_for_num_frames(
            self.bg_valid_trajectories, self.bg_can_sample, bg_batch_size)

        t1_points = torch.cat([fg_t1_points, bg_t1_points], dim=0)
        t2_points = torch.cat([fg_t2_points, bg_t2_points], dim=0)

        frames_set_t = torch.cat((t1_points[:, 2], t2_points[:, 2])).unique().int()
        source_frame_indices = torch.cat([(frames_set_t == i).nonzero() for i in t1_points[:, 2]])[:, 0]
        target_frame_indices = torch.cat([(frames_set_t == i).nonzero() for i in t2_points[:, 2]])[:, 0]

        t1_points_normalized = self.range_normalizer(t1_points, dst=self.dst_range)
        t2_points_normalized = self.range_normalizer(t2_points, dst=self.dst_range)
        t1_points[:, 2] = t1_points_normalized[:, 2]

        sample = {
            "frames_set_t": frames_set_t,
            "source_frame_indices": source_frame_indices,
            "target_frame_indices": target_frame_indices,
            "t1_points_normalized": t1_points_normalized,
            "t2_points_normalized": t2_points_normalized,
            "t1_points": t1_points,
            "target_times": t2_points[:, 2],
        }

        # Optionally add pseudo labels for GNN loss
        if self.pseudo_labels_data:
            target_frame_idx = int(t2_points[0, 2].item())
            pseudo_labels, adj_matrix = self.get_pseudo_labels_for_frame(target_frame_idx)
            sample["pseudo_labels"] = pseudo_labels[:, :2]
            sample["adj_matrix"] = adj_matrix

        return sample

