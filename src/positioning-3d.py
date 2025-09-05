"""
3D Positioning for Eye Synchrony (Demo)
Reads 50 Hz IMU & Gaze, loads stage layout of audience member seat locations, and
produces eye_in_world.csv with (x,y,z) in a center-stage/world frame.

Author: Jahanara Nares (Neurolive) — Converted to Python from MATLAB for portfolio use.

Example Run (demo data, participant B):
  python src/positioning.py --participant B --export-heading

Inputs:
  data/raw/layout/
    └─ Stage_Layout_Coordinates.csv   # must include: Participant, X_center_cm, Y_center_cm, Z_center_cm

  data/processed/demo/Participant_*
    ├─ imu_50Hz.csv                  # resampled IMU signals @50Hz
    ├─ gaze_50Hz.csv                 # resampled gaze signals @50Hz
    └─ video_timestamps.csv          # aligned video timestamps

Output (per session) in data/processed/demo/Participant_*:
    eye_in_world.csv                  # gaze mapped into 3D world coordinates (x, y, z, per sample)
    heading_50Hz.csv                  # optional debug file of heading vectors (if --export-heading is set)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter,lfilter

DT = 0.02  # 50 Hz
FOV_DEG = 90.0               # ±45° from Pupil Labs Invisible Glasses Field-of-View
EYE_W_PX = 1088              # from Pupil Labs Invisible Glasses screen resolution
EYE_H_PX = 1080              # from Pupil Labs Invisible Glasses
UP_VEC = np.array([0.0, -1.0, 0.0])  # y-up negative (upil Labs Invisible Glasses convention)
TC_DEFAULT = 10.0            # Butterworth time constant (s) → ~0.56 Hz cutoff @ 50 Hz

# ----------------------------- paths -----------------------------
REPO_ROOT      = Path(__file__).resolve().parents[1]
RAW_LAYOUT_DIR = REPO_ROOT / "data" / "raw" / "layout"
PROC_DEMO_ROOT = REPO_ROOT / "data" / "processed" / "demo"

# ----------------------------- utils -----------------------------
def butter_filt_matlab(x: np.ndarray, tc: float) -> np.ndarray:
    """
    MATLAB ButterFilt equivalent:
      fc = sqrt(2)/(2*pi*tc); [b,a] = butter(2, fc); y = filter(b,a,x)
    Note: fc here is the *normalized* digital cutoff (0..1), independent of fs,
          so behavior matches your MATLAB when DT=0.02 (50 Hz).
    """
    Wn = np.sqrt(2.0) / (2.0 * np.pi * float(tc))  # normalized (0..1), Nyquist=1
    b, a = butter(2, Wn, btype="low")
    return lfilter(b, a, np.asarray(x, dtype=float))

def rotation_step(x_angle: float, y_angle: float, z_angle: float) -> np.ndarray:
    """XYZ incremental rotation (radians)."""
    cx, cy, cz = np.cos(x_angle), np.cos(y_angle), np.cos(z_angle)
    sx, sy, sz = np.sin(x_angle), np.sin(y_angle), np.sin(z_angle)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz

def normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)

def load_layout_df() -> pd.DataFrame:
    """Load layout from MAT or CSV. CSV should include columns:
       Participant, X_center_cm, Y_center_cm, Z_center_cm (names flexible; see _getcol)."""
    csvs = list(RAW_LAYOUT_DIR.glob("Stage_Layout_Coordinates*.csv"))
    if csvs:
        return pd.read_csv(csvs[0])

    raise FileNotFoundError("No layout file found in data/raw/layout/.")

def pick_participant_row(df: pd.DataFrame, participant_id: str | None) -> pd.Series:
    # normalize Participant column
    pcol = next((c for c in df.columns if c.lower() == "participant"), None)
    if pcol is None:
        raise ValueError("Layout must have a 'Participant' column.")
    if participant_id:
        row = df[df[pcol].astype(str).str.upper() == participant_id.upper()]
        if not row.empty:
            return row.iloc[0]
        raise ValueError(f"Participant '{participant_id}' not found in layout.")
    # no id given → return first row
    return df.iloc[0]

def get_center_coords(row: pd.Series) -> tuple[float, float, float]:
    # try multiple possible names
    def _get(*names, default=None):
        for n in names:
            if n in row.index: return float(row[n])
            for c in row.index:
                if c.lower() == n.lower(): return float(row[c])
        if default is None:
            raise KeyError(f"Missing layout column (tried {names})")
        return float(default)

    Xc = _get("X Distance from center (cm) - pitch axis", default=0.0)
    Yc = _get("Y Distance from center (cm) - yaw axis", default=0.0)
    Zc = _get("Z Distance from center (cm) - roll axis", default=0.0)
    return Xc, Yc, Zc

def participant_id_from_folder(folder: Path) -> str:
    """Extract ID from folder name: 'participant_B' -> 'B'."""
    return folder.name.split("_")[-1].upper()

# ----------------------------- core per participant -----------------------------
def process_participant(p_dir: Path, layout_df: pd.DataFrame, tc: float, export_heading: bool, debug: bool):
    pid = participant_id_from_folder(p_dir)
    print(f"\n→ Positioning {p_dir.name} (ID={pid})")

    imu_path  = p_dir / "imu_50Hz.csv"
    gaze_path = p_dir / "gaze_50Hz.csv"
    if not imu_path.exists() or not gaze_path.exists():
        print("  • Missing imu_50Hz.csv or gaze_50Hz.csv — skipping.")
        return

    imu  = pd.read_csv(imu_path)
    gaze = pd.read_csv(gaze_path)

    # --- Gyros
    gx = pd.to_numeric(imu.get("gyro x [deg/s]"), errors="coerce").to_numpy()
    gy = pd.to_numeric(imu.get("gyro y [deg/s]"), errors="coerce").to_numpy()
    gz = pd.to_numeric(imu.get("gyro z [deg/s]"), errors="coerce").to_numpy()

    # Filter (MATLAB-style), integrate, drift-correct
    gx_f = butter_filt_matlab(gx, tc)
    gy_f = butter_filt_matlab(gy, tc)
    gz_f = butter_filt_matlab(gz, tc)

    t = np.arange(len(gx_f)) * DT
    pitch_f = np.cumsum(gx_f * DT)   # deg
    yaw_f   = np.cumsum(gy_f * DT)
    roll_f  = np.cumsum(gz_f * DT)

    bx = np.polyfit(t, pitch_f, 1)[0]
    by = np.polyfit(t, yaw_f,   1)[0]
    bz = np.polyfit(t, roll_f,  1)[0]

    gx_cf = gx_f - bx
    gy_cf = gy_f - by
    gz_cf = gz_f - bz

    # Heading via incremental rotations
    n = len(gx_cf)
    heading = np.zeros((n, 3), dtype=float)
    heading[0] = np.array([0.0, 0.0, 1.0])
    R_total = np.eye(3)
    for i in range(1, n):
        xa = np.deg2rad(gx_cf[i]) * DT
        ya = np.deg2rad(gy_cf[i]) * DT
        za = np.deg2rad(gz_cf[i]) * DT
        R_step = rotation_step(xa, ya, za)
        R_total = R_step @ R_total
        heading[i] = R_total @ np.array([0.0, 0.0, 1.0])
    heading = normalize_rows(heading)

    # --- Gaze normalization (pixels -> [-1,1])
    gx_px = pd.to_numeric(gaze.get("gaze x [px]"), errors="coerce").to_numpy()
    gy_px = pd.to_numeric(gaze.get("gaze y [px]"), errors="coerce").to_numpy()
    x_norm = 2.0 * gx_px / EYE_W_PX - 1.0
    y_norm = 2.0 * gy_px / EYE_H_PX - 1.0

    # Optional: filter the normalized gaze (matches MATLAB ButterFilt)
    x_nf = butter_filt_matlab(x_norm, tc)
    y_nf = butter_filt_matlab(y_norm, tc)

    # --- Layout row for this participant
    row = pick_participant_row(layout_df, pid)
    Xc_cm, Yc_cm, Zc_cm = get_center_coords(row)

    # --- Map normalized gaze to cm at distance Z
    half = np.deg2rad(FOV_DEG / 2.0)
    width_cm  = 2.0 * abs(Zc_cm) * np.tan(half)
    height_cm = 2.0 * abs(Zc_cm) * np.tan(half)
    gaze_x_cm = x_nf * (width_cm / 2.0)
    gaze_y_cm = y_nf * (height_cm / 2.0)

    # --- Transform to world frame
    L = min(len(heading), len(gaze_x_cm))
    xw = np.zeros(L); yw = np.zeros(L); zw = np.zeros(L)
    for i in range(L):
        h = heading[i]
        r = np.cross(UP_VEC, h)
        rn = np.linalg.norm(r) or 1.0
        r = r / rn
        up_ortho = np.cross(h, r)
        Rb = np.vstack([r, up_ortho, h])  # rows: right, up, forward

        eye_vec = np.array([gaze_x_cm[i], gaze_y_cm[i], -Zc_cm])
        world_vec = Rb.T @ eye_vec

        xw[i] = world_vec[0] - Xc_cm
        yw[i] = world_vec[1] + Yc_cm
        zw[i] = world_vec[2]

    # --- Write outputs into the participant folder
    eye_world_out = p_dir / "eye_in_world.csv"
    out = pd.DataFrame({
        "gaze_x_norm": x_norm[:L],
        "gaze_y_norm": y_norm[:L],
        "x": xw, "y": yw, "z": zw
    })
    out.to_csv(eye_world_out, index=False)
    print(f"  ✓ Wrote {eye_world_out}")

    if export_heading:
        heading_out = p_dir / "heading_50Hz.csv"
        pd.DataFrame(heading, columns=["hx","hy","hz"]).to_csv(heading_out, index=False)
        print(f"  • Wrote {heading_out}")

    if debug:
        print(f"  [debug] drift deg/s: bx={bx:.4g}, by={by:.4g}, bz={bz:.4g}")
        print(f"  [debug] world x[{xw.min():.2f},{xw.max():.2f}] "
              f"y[{yw.min():.2f},{yw.max():.2f}] z[{zw.min():.2f},{zw.max():.2f}]")

# ----------------------------- entrypoint -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", type=float, default=TC_DEFAULT, help="Butterworth time constant (s). Default 10.")
    ap.add_argument("--export-heading", action="store_true", help="Write heading_50Hz.csv per participant.")
    ap.add_argument("--debug", action="store_true", help="Enable debug printing")
    args = ap.parse_args()

    layout_df = load_layout_df()

    # Loop over all participant folders
    for p_dir in sorted(PROC_DEMO_ROOT.iterdir()):
        if p_dir.is_dir() and p_dir.name.lower().startswith("participant_"):
            process_participant(p_dir, layout_df, tc=args.tc, export_heading=args.export_heading, debug=args.debug)


if __name__ == "__main__":
    main()
