import os
import subprocess
import sys
import cv2

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pose_pipeline.py <name>")
        sys.exit(1)

    name = sys.argv[1]

    # Define paths
    frames_dir = f"/home/adminubuntu/Lumenova/datasets/{name}"
    json_input_path = f"/home/adminubuntu/Lumenova/PoseAnything/readyoutputs/{name}/pose_results.json"
    output_root = "/home/adminubuntu/Lumenova/output"
    output_dir = os.path.join(output_root, f"{name}_output")
    os.makedirs(output_dir, exist_ok=True)

    # Automatically detect resolution from first frame
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
    if not frame_files:
        raise ValueError(f"No image frames found in {frames_dir}")
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    img = cv2.imread(first_frame_path)
    if img is None:
        raise ValueError(f"Could not read first frame: {first_frame_path}")
    orig_h, orig_w = img.shape[:2]

    # Run adjust_json.py
    print("✅ Running adjust_json.py...")
    adjust_cmd = [
        "python", "/home/adminubuntu/Lumenova/scripts/adjust_json.py",
        "--json-path", json_input_path,
        "--orig-w", str(orig_w),
        "--orig-h", str(orig_h)
    ]
    subprocess.run(adjust_cmd, check=True, cwd=output_dir)

    # Paths after adjustment
    adjusted_json_path = os.path.join(output_dir, "pose_results_adjusted.json")

    # Run create_video_from_pose.py
    print("🎥 Running create_video_from_pose.py...")
    create_video_cmd = [
        "python", "/home/adminubuntu/Lumenova/scripts/create_video_from_pose.py",
        "--frames-dir", frames_dir,
        "--json-path", adjusted_json_path,
        "--output-dir", os.path.join(output_dir, "annotated_frames"),
        "--output-video", os.path.join(output_dir, f"{name}_output_from_pose.mp4")
    ]
    subprocess.run(create_video_cmd, check=True)

    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
