"""
Data Pre-Processing for Eye Synchrony
Resample IMU and Gaze data to 50 Hz and align streams.

Author: Jahanara Nares (Neurolive) — Converted to Python from MATLAB for portfolio use.

Inputs:
  data/raw/demo/Participant_*
    ├─ imu.csv                  # columns include: timestamp [ns], gyro x/y/z [deg/s], acceleration x/y/z [g], roll/pitch/yaw [deg]
    ├─ gaze.csv                 # columns include: timestamp [ns], gaze x [px], gaze y [px]
    └─ world_timestamps.csv     # columns include: timestamp [ns]  of video recording

Output (per session) in data/processed/demo:
    imu_50Hz.csv, gaze_50Hz.csv, video_timestamps.csv
"""

import pandas as pd
from pathlib import Path

# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT  = REPO_ROOT / "data" / "raw" / "demo"
PROC_ROOT = REPO_ROOT / "data" / "processed" / "demo"
PROC_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_RATE_HZ = 50.0   # 50 Hz
PERIOD = pd.to_timedelta(1.0 / TARGET_RATE_HZ, unit="s")  # 20 ms

TARGET_RATE_HZ = 50.0   # 50 Hz
PERIOD = pd.to_timedelta(1.0 / TARGET_RATE_HZ, unit="s")  # 20 ms

# ---- Required columns ----
IMU_COLS = [
    "gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]",
    "acceleration x [g]", "acceleration y [g]", "acceleration z [g]",
    "roll [deg]", "pitch [deg]", "yaw [deg]"
]
GAZE_COLS = ["gaze x [px]", "gaze y [px]"]


def read_and_resample(csv_path: Path, value_cols: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "timestamp [ns]" not in df.columns:
        raise ValueError(f"'timestamp [ns]' column not found in {csv_path}")
    # coerce timestamps and values
    ts = pd.to_datetime(df["timestamp [ns]"], unit="ns", errors="coerce")
    vals = df[value_cols].apply(pd.to_numeric, errors="coerce")
    # index and resample
    x = vals.copy()
    x["timestamp [datetime]"] = ts
    x = x.dropna(subset=["timestamp [datetime]"]).set_index("timestamp [datetime]").sort_index()
    x = x.resample(PERIOD).mean()
    return x


def process_participant(p_dir: Path):
    print(f"\n→ Processing {p_dir.name}")
    proc_dir = PROC_ROOT / p_dir.name
    proc_dir.mkdir(parents=True, exist_ok=True)

    imu_in   = p_dir / "demo_imu.csv"
    gaze_in  = p_dir / "demo_gaze.csv"
    video_in = p_dir / "demo_world_timestamps.csv"

    imu_out   = proc_dir / "imu_50Hz.csv"
    gaze_out  = proc_dir / "gaze_50Hz.csv"
    video_out = proc_dir / "video_timestamps.csv"

    # IMU & Gaze -> resampled @ 50 Hz
    imu_50  = read_and_resample(imu_in, IMU_COLS)
    gaze_50 = read_and_resample(gaze_in, GAZE_COLS)

    # Align start (use the later of the two starts)
    start_time = max(imu_50.index.min(), gaze_50.index.min())
    imu_50  = imu_50[imu_50.index  >= start_time]
    gaze_50 = gaze_50[gaze_50.index >= start_time]

    # Save outputs
    imu_50.to_csv(imu_out)
    gaze_50.to_csv(gaze_out)
    print(f"✓ Wrote {imu_out} and {gaze_out}")

    # Video timestamps -> datetime format
    if video_in.exists():
        v = pd.read_csv(video_in)
        if "timestamp [ns]" in v.columns:
            v["timestamp [datetime]"] = pd.to_datetime(v["timestamp [ns]"], unit="ns", errors="coerce")
        v.to_csv(video_out, index=False)
        print(f"✓ Wrote {video_out}")
    else:
        print("• No world_timestamps.csv found; skipped video timestamps.")

def main():
    for p_dir in RAW_ROOT.iterdir():
        if p_dir.is_dir() and p_dir.name.startswith("Participant_"):
            process_participant(p_dir)

if __name__ == "__main__":
    main()