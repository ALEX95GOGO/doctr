import cv2
import numpy as np
import csv
import os

def save_frame_time_stats_to_csv(video_path, csv_path, nominal_fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Get nominal FPS from video if not provided
    if nominal_fps is None:
        nominal_fps = cap.get(cv2.CAP_PROP_FPS)
        if nominal_fps <= 0:
            cap.release()
            raise RuntimeError(
                "Could not determine nominal FPS from video. "
                "Pass nominal_fps explicitly to save_frame_time_stats_to_csv()."
            )

    timestamps = []  # seconds

    # Read all frames and record timestamps (from container)
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamps.append(t_sec)

    cap.release()

    if len(timestamps) < 2:
        raise RuntimeError("Not enough frames to compute timings.")

    timestamps = np.array(timestamps)

    # Time between consecutive frames
    frame_intervals = np.diff(timestamps)  # seconds

    # Expected interval and jitter
    expected_interval = 1.0 / nominal_fps
    jitter_ms = (frame_intervals - expected_interval) * 1000.0

    # Actual FPS for each interval (frame rate between frame i and i+1)
    actual_fps = 1.0 / frame_intervals

    # Frame indices for intervals: between i and i+1
    frame_indices_start = np.arange(0, len(timestamps) - 1)
    frame_indices_end = frame_indices_start + 1

    # Write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "frame_start",
            "frame_end",
            "timestamp_start_sec",
            "timestamp_end_sec",
            "interval_sec",
            "nominal_fps",
            "expected_interval_sec",
            "actual_fps",
            "jitter_ms"
        ])

        for i in range(len(frame_intervals)):
            writer.writerow([
                int(frame_indices_start[i]),
                int(frame_indices_end[i]),
                float(timestamps[i]),
                float(timestamps[i + 1]),
                float(frame_intervals[i]),
                float(nominal_fps),
                float(expected_interval),
                float(actual_fps[i]),
                float(jitter_ms[i]),
            ])

    print(f"Saved frame time stats to: {csv_path}")


if __name__ == "__main__":
    video_path = "/projects/HAT/Zhuoli/lht/05_Videos_Study5/A001.mp4"
    csv_path = "/projects/HAT/Zhuoli/lht/04_Study4_Gameplay_Videos/F001_frame_time_stats.csv"

    save_frame_time_stats_to_csv(video_path, csv_path)

