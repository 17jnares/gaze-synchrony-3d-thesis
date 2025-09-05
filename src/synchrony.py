"""
Windowed Synchrony - Pairwise & Individual-to-Group (Demo)

- Assumes 50 Hz signals from positioning stage: data/processed/demo/Participant_*/eye_in_world.csv
- Uses x and y world-coordinates; computes windowed Pearson correlations.
- Pairwise: Fisher-z average of (r_x, r_y) per pair; group mean = mean of all r_x and r_y across pairs.
- Indiv-to-Group: Fisher-z average of (corr(self.x, mean_others.x), corr(self.y, mean_others.y)).

Input:
  data/processed/demo/
    └─ Participant_*
        └─ eye_in_world.csv     # gaze mapped into 3D world coordinates (x, y, z, per sample)

Outputs (per session) in data/processed/demo/:
    data/processed/demo/
    ├── pairwise_sync/
    │   ├── 1s/
    │   │   └── pairwise_group_sync.csv
    │   └── 5s/
    │       └── pairwise_group_sync.csv
    └── indiv_to_group_sync/
        ├── 1s/
        │   └── indiv_vs_group_sync.csv
        └── 5s/
            └── indiv_vs_group_sync.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr

FS = 50.0           # Hz
DT = 1.0 / FS
WINDOW_SEC_LIST = [1.0, 5.0]  # compute 1s and 5s time-scales of synchrony

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC_DEMO_ROOT = REPO_ROOT / "data" / "processed" / "demo"

def participant_id_from_folder(folder: Path) -> str:
    # "participant_B" -> "B"
    return folder.name.split("_")[-1].upper()

def safe_read_eye_in_world(p_dir: Path) -> pd.DataFrame | None:
    f = p_dir / "eye_in_world.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    need = {"x", "y"}
    if not need.issubset({c.lower() for c in df.columns}):
        # try to normalize headers
        df.columns = [c.strip().lower() for c in df.columns]
        if not need.issubset(set(df.columns)):
            raise ValueError(f"{f} missing required columns 'x' and 'y'. Have {list(df.columns)}")
    return df

def fisher_z(r: float) -> float:
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inv_fisher_z(z: float) -> float:
    return np.tanh(z)

def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan, np.nan
    try:
        r, p = pearsonr(a[mask], b[mask])
    except Exception:
        r, p = np.nan, np.nan
    return float(r), float(p)

def window_indices(n: int, win_len: int):
    """Yield (start, end) for non-overlapping windows that fit fully."""
    n_windows = n // win_len
    for w in range(n_windows):
        s = w * win_len
        e = s + win_len
        yield w, s, e
    return n_windows

def compute_for_window(X_win: np.ndarray, Y_win: np.ndarray, ids: list[str]):
    """
    X_win/Y_win: shape (N, W) for this window; ids: list of participant IDs
    Returns:
      pair_labels: ["PairID_A_B", ...]
      r_avg_pairs: vector len=P (Fisher-z avg of x/y per pair)
      r_x_pairs, r_y_pairs: vectors len=P (for group mean)
      indiv_vs_group: vector len=N (Fisher-z avg vs. group-minus-self)
    """
    N, W = X_win.shape
    pairs = list(combinations(range(N), 2))
    pair_labels = [f"PairID_{ids[i]}_{ids[j]}" for i, j in pairs]

    r_x_pairs = np.full(len(pairs), np.nan)
    r_y_pairs = np.full(len(pairs), np.nan)
    r_avg_pairs = np.full(len(pairs), np.nan)

    # Pairwise
    for k, (i, j) in enumerate(pairs):
        r_x, _ = safe_pearson(X_win[i], X_win[j])
        r_y, _ = safe_pearson(Y_win[i], Y_win[j])
        r_x_pairs[k] = r_x
        r_y_pairs[k] = r_y
        # Fisher-z average across axes
        z_mean = np.nanmean([fisher_z(r_x), fisher_z(r_y)])
        r_avg_pairs[k] = inv_fisher_z(z_mean)

    # Individual vs group-minus-self
    indiv_vs_group = np.full(N, np.nan)
    for i in range(N):
        others = [k for k in range(N) if k != i]
        x_group = np.nanmean(X_win[others, :], axis=0)
        y_group = np.nanmean(Y_win[others, :], axis=0)
        r_x, _ = safe_pearson(X_win[i], x_group)
        r_y, _ = safe_pearson(Y_win[i], y_group)
        z_mean = np.nanmean([fisher_z(r_x), fisher_z(r_y)])
        indiv_vs_group[i] = inv_fisher_z(z_mean)

    return pair_labels, r_avg_pairs, r_x_pairs, r_y_pairs, indiv_vs_group

def main():
    # --- discover participants
    print(PROC_DEMO_ROOT)
    p_dirs = sorted([d for d in PROC_DEMO_ROOT.iterdir() if d.is_dir() and d.name.lower().startswith("participant_")])
    if len(p_dirs) < 2:
        raise RuntimeError("Need at least 2 participants under data/processed/demo/participant_*/")

    # load all eye_in_world
    participants = []
    for d in p_dirs:
        df = safe_read_eye_in_world(d)
        if df is None:
            continue
        pid = participant_id_from_folder(d)
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
        participants.append((pid, d, x, y))

    if len(participants) < 2:
        raise RuntimeError("Found fewer than 2 valid eye_in_world.csv files.")

    # align to common length
    min_len = min(len(x) for (_, _, x, y) in participants)
    ids = [pid for (pid, _, _, _) in participants]
    X = np.vstack([x[:min_len] for (_, _, x, _) in participants])
    Y = np.vstack([y[:min_len] for (_, _, _, y) in participants])

    for window_sec in WINDOW_SEC_LIST:
        win_len = int(round(window_sec * FS))
        n_windows = min_len // win_len
        if n_windows < 1:
            print(f"• Not enough samples for window {window_sec}s — skipping.")
            continue

        # Prepare outputs
        pair_cols = None
        pair_ravg_matrix = []  # shape (n_windows, num_pairs)
        group_mean_series = [] # len n_windows
        indiv_cols = [f"PartID_{pid}" for pid in ids]
        indiv_matrix = []      # shape (n_windows, N)

        # Iterate windows
        for w, s, e in window_indices(min_len, win_len):
            Xw = X[:, s:e]
            Yw = Y[:, s:e]

            pair_labels, r_avg_pairs, r_x_pairs, r_y_pairs, indiv_vs_group = compute_for_window(Xw, Yw, ids)
            if pair_cols is None:  # first window defines order
                pair_cols = pair_labels

            pair_ravg_matrix.append(r_avg_pairs)
            # Group avg across pairs and axes (match MATLAB: mean of concatenated r_x & r_y across pairs)
            group_mean = np.nanmean(np.concatenate([r_x_pairs, r_y_pairs]))
            group_mean_series.append(group_mean)

            indiv_matrix.append(indiv_vs_group)

        # Build DataFrames
        time_s = np.arange(n_windows) * window_sec  # window start times

        # Pairwise output
        df_pair = pd.DataFrame({"time_s": time_s})
        pair_ravg_matrix = np.vstack(pair_ravg_matrix)  # (n_windows, num_pairs)
        for j, col in enumerate(pair_cols):
            df_pair[col] = pair_ravg_matrix[:, j]
        df_pair["GroupAverage"] = np.asarray(group_mean_series, float)

        # Indiv→Group output
        df_indiv = pd.DataFrame({"time_s": time_s})
        indiv_matrix = np.vstack(indiv_matrix)  # (n_windows, N)
        for j, col in enumerate(indiv_cols):
            df_indiv[col] = indiv_matrix[:, j]

        # Save
        out_pair_dir  = PROC_DEMO_ROOT / "pairwise_sync" / f"{int(window_sec)}s"
        out_indiv_dir = PROC_DEMO_ROOT / "indiv_to_group_sync" / f"{int(window_sec)}s"
        out_pair_dir.mkdir(parents=True, exist_ok=True)
        out_indiv_dir.mkdir(parents=True, exist_ok=True)

        pair_path  = out_pair_dir  / "pairwise_group_sync.csv"
        indiv_path = out_indiv_dir / "indiv_vs_group_sync.csv"
        df_pair.to_csv(pair_path, index=False)
        df_indiv.to_csv(indiv_path, index=False)

        print(f"✓ Saved pairwise sync ({int(window_sec)}s): {pair_path}")
        print(f"✓ Saved indiv→group sync ({int(window_sec)}s): {indiv_path}")

if __name__ == "__main__":
    main()