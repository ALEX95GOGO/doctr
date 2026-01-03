import cv2
import os
import numpy as np
import argparse

# ==== CONFIGURATION ====
# Define the region of interest (ROI)
roi_x, roi_y = 600, 1150
roi_w, roi_h = 1400, 300
# =======================

save_dir = "saved_frames"
os.makedirs(save_dir, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Video ROI frame extractor")

    parser.add_argument(
        "--base_path",
        type=str,
        default="/projects/HAT/Zhuoli/lht/05_Videos_Study5",
        help="Base directory of the video file"
    )

    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Video filename (e.g., A001.mp4)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Build the full video path
    video_path = (
        args.file_name
        if args.base_path == ""
        else os.path.join(args.base_path, args.file_name)
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video:", video_path)
        return

    prev_result = None
    frame_count = 0
    diff_threshold = 5000   # adjust for sensitivity

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        h, w = frame.shape[:2]
        x1 = max(0, roi_x)
        y1 = max(0, roi_y)
        x2 = min(w, roi_x + roi_w)
        y2 = min(h, roi_y + roi_h)

        if x1 >= x2 or y1 >= y2:
            print("Invalid ROI for current frame size.")
            break

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # White threshold
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Save frame
        filename = f"{save_dir}/frame_{frame_count:06d}.png"
        cv2.imwrite(filename, result)
        #print("Saved:", filename)

        prev_result = result.copy()
        frame_count += 1

        cv2.imshow("ROI", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

