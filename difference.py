import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "Walker_Data/csv"
OUTPUT_DIR = "Walker_Data/output_difference"

# ===== 図サイズ設定 =====
FIG_WIDTH = 9
FIG_HEIGHT = 6
# ========================

# ===== 縦軸上限設定 =====
LEFT_Y_MAX = 1800     # None にすれば自動
RIGHT_Y_MAX = 180     # None にすれば自動
# =======================

os.makedirs(OUTPUT_DIR, exist_ok=True)

required = [
    "elapsed_time",
    "player_posx", "player_posy", "player_rotz",
    "target_posx", "target_posy", "target_rotz",
]

def extract_round_from_filename(filename: str) -> str:
    match = re.search(r'R(\d+)', filename)
    if match:
        return f"Round {match.group(1)}"
    return "Round ?"

def make_plot_to_pdf(csv_path: str, out_pdf_path: str) -> None:
    df = pd.read_csv(csv_path)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(csv_path)} に必要な列がありません: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 位置差分 ---
    dx = df["target_posx"] - df["player_posx"]
    dy = df["target_posy"] - df["player_posy"]

    df["xdiff_abs"] = dx.abs()
    df["ydiff_abs"] = dy.abs()
    df["distance"] = np.sqrt(dx**2 + dy**2)

    # --- rotz差分 ---
    raw_rotz_diff = df["target_rotz"] - df["player_rotz"]
    wrapped = (raw_rotz_diff + 180) % 360 - 180
    df["rotz_diff_abs"] = wrapped.abs()

    plot_cols = [
        "elapsed_time",
        "xdiff_abs",
        "ydiff_abs",
        "distance",
        "rotz_diff_abs",
    ]

    dpp = df[plot_cols].dropna()
    if len(dpp) == 0:
        raise ValueError(f"{os.path.basename(csv_path)} は有効な行がありません")

    t = dpp["elapsed_time"].to_numpy()
    total_time = t.max()

    # ===== 図サイズ反映 =====
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # --- 左軸 ---
    l1, = ax1.plot(t, dpp["xdiff_abs"], label="|xdiff|", color="orange")
    l2, = ax1.plot(t, dpp["ydiff_abs"], label="|ydiff|", color="blue")
    l3, = ax1.plot(t, dpp["distance"], label="distance", color="red")

    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("position diff (abs) / distance")
    ax1.grid(True)

    if LEFT_Y_MAX is not None:
        ax1.set_ylim(0, LEFT_Y_MAX)

    # --- 右軸 ---
    ax2 = ax1.twinx()
    l4, = ax2.plot(t, dpp["rotz_diff_abs"], label="|rotz_diff| (deg)", color="green")
    ax2.set_ylabel("rotz diff abs (deg)")

    if RIGHT_Y_MAX is not None:
        ax2.set_ylim(0, RIGHT_Y_MAX)

    # --- 凡例 ---
    lines = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center right")

    # --- タイトル ---
    filename = os.path.basename(csv_path)
    round_text = extract_round_from_filename(filename)
    title_text = f"{round_text} (time: {total_time:.1f}s)"

    ax1.text(
        0.98, 0.98,
        title_text,
        transform=ax1.transAxes,
        fontsize=18,
        fontweight='bold',
        horizontalalignment='right',
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
    )

    plt.tight_layout()
    fig.savefig(out_pdf_path, format="pdf")
    plt.close(fig)

def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"{INPUT_DIR}/ にCSVが見つかりません")

    ok = 0
    ng = 0

    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_pdf = os.path.join(OUTPUT_DIR, f"{base}.pdf")

        try:
            make_plot_to_pdf(csv_path, out_pdf)
            print(f"[OK] {csv_path} -> {out_pdf}")
            ok += 1
        except Exception as e:
            print(f"[NG] {csv_path}: {e}")
            ng += 1

    print(f"done．OK={ok}, NG={ng}")

if __name__ == "__main__":
    main()
