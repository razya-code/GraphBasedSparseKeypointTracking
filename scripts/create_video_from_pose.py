import cv2
import json
import numpy as np
import os
import subprocess
import argparse
from natsort import natsorted

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

def generate_vivid_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return colors

def global_scale_point(x_norm, y_norm, scale=1):
    center_x, center_y = 0.5, 0.5
    x_scaled = center_x + (x_norm - center_x) * scale
    y_scaled = center_y + (y_norm - center_y) * scale
    x_scaled = min(max(x_scaled, 0), 1)
    y_scaled = min(max(y_scaled, 0), 1)
    return x_scaled, y_scaled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory containing frames")
    parser.add_argument("--json-path", type=str, required=True, help="Path to adjusted JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for annotated frames")
    parser.add_argument("--output-video", type=str, required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading JSON from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    queries = data['queries']
    print(f"Loaded {len(queries)} query frames from JSON.")

    print("🔧 Smoothing keypoints in time...")
    queries = smooth_keypoints(queries, window_size=5)
    print("✅ Keypoints smoothed.")

    frame_files = natsorted([f for f in os.listdir(args.frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
    print(f"Found {len(frame_files)} frame images in {args.frames_dir}")

    first_frame = cv2.imread(os.path.join(args.frames_dir, frame_files[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")

    height, width = first_frame.shape[:2]
    print(f"Frame size detected: {width}x{height}")

    num_points = len(queries[0]['keypoints'])
    point_colors = generate_vivid_colors(num_points)

    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(args.frames_dir, frame_file)
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

            overlay = frame.copy()

            for i in range(len(pts)):
                for j in range(len(pts)):
                    if adj_matrix[i][j]:
                        cv2.line(overlay, pts[i], pts[j], (0, 0, 255), 4)

            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            for idx, (x, y) in enumerate(pts):
                cv2.circle(frame, (x, y), 8, point_colors[idx], -1)
        else:
            print(f"ℹ️ Frame {frame_idx} has no keypoints in JSON, skipping drawing.")

        output_frame_path = os.path.join(args.output_dir, frame_file)
        cv2.imwrite(output_frame_path, frame)
        print(f"✅ Saved annotated frame {frame_idx + 1}/{len(frame_files)}: {output_frame_path}")

    print(f"🎥 Creating video from frames using ffmpeg...")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(args.fps),
        "-pattern_type", "glob",
        "-i", os.path.join(args.output_dir, "*.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        args.output_video
    ]
    print(f"Running: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

    print(f"✅ Video created: {args.output_video}")
    print("🎉 All done!")

if __name__ == "__main__":
    main()
