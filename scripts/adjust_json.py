import json
import numpy as np
import argparse
import os

def adjust_keypoints_from_square(keypoints, orig_w, orig_h, square_size):
    adjusted = []

    # Determine pad and final padded size
    if orig_w > orig_h:
        pad = (orig_w - orig_h) / 2
        padded_size = orig_w
        scale = square_size / padded_size
        for kp in keypoints:
            x_norm, y_norm = kp
            x_px = x_norm * square_size
            y_px = y_norm * square_size

            # Remove vertical padding
            y_content_px = (y_px - pad * scale) / scale
            x_content_px = x_px / scale

            # Clip to avoid going negative or overshooting
            x_content_px = np.clip(x_content_px, 0, orig_w)
            y_content_px = np.clip(y_content_px, 0, orig_h)

            # Normalize
            x_final = x_content_px / orig_w
            y_final = y_content_px / orig_h

            adjusted.append([x_final, y_final])
    elif orig_h > orig_w:
        pad = (orig_h - orig_w) / 2
        padded_size = orig_h
        scale = square_size / padded_size
        for kp in keypoints:
            x_norm, y_norm = kp
            x_px = x_norm * square_size
            y_px = y_norm * square_size

            # Remove horizontal padding
            x_content_px = (x_px - pad * scale) / scale
            y_content_px = y_px / scale

            x_content_px = np.clip(x_content_px, 0, orig_w)
            y_content_px = np.clip(y_content_px, 0, orig_h)

            x_final = x_content_px / orig_w
            y_final = y_content_px / orig_h

            adjusted.append([x_final, y_final])
    else:
        scale = square_size / orig_w  # or orig_h (they are equal here)
        for kp in keypoints:
            x_norm, y_norm = kp
            x_px = x_norm * square_size
            y_px = y_norm * square_size

            x_content_px = x_px / scale
            y_content_px = y_px / scale

            x_content_px = np.clip(x_content_px, 0, orig_w)
            y_content_px = np.clip(y_content_px, 0, orig_h)

            x_final = x_content_px / orig_w
            y_final = y_content_px / orig_h

            adjusted.append([x_final, y_final])
    return adjusted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True, help="Path to input JSON file (pose_results.json)")
    parser.add_argument("--orig-w", type=int, required=True, help="Original image width")
    parser.add_argument("--orig-h", type=int, required=True, help="Original image height")
    parser.add_argument("--square-size", type=int, default=369, help="Square image size used by PoseAnything")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    queries = data["queries"]

    for entry in queries:
        keypoints = entry["keypoints"]
        adjusted_kps = adjust_keypoints_from_square(keypoints, args.orig_w, args.orig_h, args.square_size)
        entry["keypoints"] = adjusted_kps

    # Adjust support keypoints too
    support_keypoints = data["support"]["keypoints"]
    adjusted_support_kps = adjust_keypoints_from_square(support_keypoints, args.orig_w, args.orig_h, args.square_size)
    data["support"]["keypoints"] = adjusted_support_kps

    output_path = os.path.join(os.getcwd(), "pose_results_adjusted.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Adjusted JSON saved to: {output_path}")

if __name__ == "__main__":
    main()
