import cv2
import json
import numpy as np
import os
import subprocess
from natsort import natsorted

# === Configuration ===
frames_dir = '/home/adminubuntu/Lumenova/dino-tracker/dataset/horsejump/video'
json_path = '/home/adminubuntu/Lumenova/PoseAnything/output/pose_results.json'
output_dir = '/home/adminubuntu/Lumenova/PoseAnything/output/annotated_frames'
output_video = '/home/adminubuntu/Lumenova/PoseAnything/output/output_with_graph.mp4'
fps = 10

point_radius = 8
edge_thickness = 4
edge_alpha = 0.5  # Transparency for edges
trajectory_length = 20  # Number of past positions to show

# === Load JSON data ===
print(f"Loading JSON from {json_path}...")
with open(json_path, 'r') as f:
    data = json.load(f)

queries = data['queries']
print(f"Loaded {len(queries)} query frames from JSON.")

# === Smooth keypoints in time ===
def smooth_keypoints(queries, window_size=5):
    half_window = window_size // 2
    num_frames = len(queries)
    num_points = len(queries[0]['keypoints'])
    kp_array = np.zeros((num_frames, num_points, 2), dtype=np.float32)

    for i, q in enumerate(queries):
        kp_array[i] = np.array(q['keypoints'])

    kp_smoothed = np.copy(kp_array)
    for t in range(num_frames):
        start = max(0, t - half_window)
        end = min(num_frames, t + half_window + 1)
        kp_smoothed[t] = np.mean(kp_array[start:end], axis=0)

    for i in range(num_frames):
        queries[i]['keypoints'] = kp_smoothed[i].tolist()

    return queries

print("🔧 Smoothing keypoints in time...")
queries = smooth_keypoints(queries, window_size=5)
print("✅ Keypoints smoothed.")

# === Generate vivid color palette (HSV-based) ===
def generate_vivid_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return colors

# === Global scaling function ===
def global_scale_point(x_norm, y_norm, scale=1):
    center_x, center_y = 0.5, 0.5
    x_scaled = center_x + (x_norm - center_x) * scale
    y_scaled = center_y + (y_norm - center_y) * scale
    x_scaled = min(max(x_scaled, 0), 1)
    y_scaled = min(max(y_scaled, 0), 1)
    return x_scaled, y_scaled

# === Prepare output directory ===
os.makedirs(output_dir, exist_ok=True)

# === Get sorted frame files ===
frame_files = natsorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
print(f"Found {len(frame_files)} frame images in {frames_dir}")

# === Get frame size ===
first_frame_path = os.path.join(frames_dir, frame_files[0])
first_frame = cv2.imread(first_frame_path)
if first_frame is None:
    raise ValueError(f"Could not read first frame: {first_frame_path}")

height, width = first_frame.shape[:2]
print(f"Frame size detected: {width}x{height}")

# === Prepare point colors ===
num_points = len(queries[0]['keypoints'])
point_colors = generate_vivid_colors(num_points)

# === Initialize trajectory buffers ===
trajectories = [[] for _ in range(num_points)]

# === Process frames ===
for frame_idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"⚠️ Warning: Could not read {frame_file}, skipping.")
        continue

    if frame.shape[1] != width or frame.shape[0] != height:
        print(f"⚠️ Size mismatch at {frame_file}: expected ({width}, {height}), got ({frame.shape[1]}, {frame.shape[0]}). Skipping.")
        continue

    query = next((q for q in queries if q['frame'] == frame_idx), None)
    if query is not None:
        keypoints = np.array(query['keypoints'])
        adj_matrix = np.array(query['adjacency_matrix'])

        pts = []
        for (x_norm, y_norm) in keypoints:
            x_s, y_s = global_scale_point(x_norm, y_norm, scale=1.1)
            x = int(x_s * width)
            y = int(y_s * height)
            pts.append((x, y))

        # Update trajectory buffers
        for idx, pt in enumerate(pts):
            trajectories[idx].append(pt)
            if len(trajectories[idx]) > trajectory_length:
                trajectories[idx].pop(0)

        # Create overlay for transparent edges
        overlay = frame.copy()

        # Draw edges on overlay
        num_pts = len(pts)
        for i in range(num_pts):
            for j in range(num_pts):
                if adj_matrix[i][j]:
                    cv2.line(overlay, pts[i], pts[j], (0, 0, 255), edge_thickness)

        # Blend overlay with frame
        cv2.addWeighted(overlay, edge_alpha, frame, 1 - edge_alpha, 0, frame)

        # Draw points on final frame
        for idx, (x, y) in enumerate(pts):
            cv2.circle(frame, (x, y), point_radius, point_colors[idx], -1)

        # Draw trajectory lines
        for idx, trail in enumerate(trajectories):
            if len(trail) > 1:
                pts_array = np.array(trail, dtype=np.int32)
                cv2.polylines(frame, [pts_array], isClosed=False, color=point_colors[idx], thickness=2)
    else:
        print(f"ℹ️ Frame {frame_idx} has no keypoints in JSON, skipping drawing.")

    output_frame_path = os.path.join(output_dir, frame_file)
    cv2.imwrite(output_frame_path, frame)
    print(f"✅ Saved annotated frame {frame_idx + 1}/{len(frame_files)}: {output_frame_path}")

# === Create video using ffmpeg ===
print(f"🎥 Creating video from frames using ffmpeg...")

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-framerate", str(fps),
    "-pattern_type", "glob",
    "-i", os.path.join(output_dir, "*.jpg"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    output_video
]

print(f"Running: {' '.join(ffmpeg_cmd)}")
subprocess.run(ffmpeg_cmd, check=True)

print(f"✅ Video created: {output_video}")
print("🎉 All done! Points are now vivid, edges are transparent, and smooth trajectories are shown.")
