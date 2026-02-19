import os
import glob
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INPUT_DIR = "Walker_Data/csv"
OUTPUT_DIR = "Walker_Data/output_trajectory"

# 固定軸範囲
X_LIM = (-1000, 1000)
Y_LIM = (-600, 600)

# 線
LINE_WIDTH = 1.2

# 色（HEX指定）
PLAYER_COLOR = "#0000FF"   # 青
TARGET_COLOR = "#FF0000"   # 赤

# 三角（鋭角タイプ：自前ポリゴン方式）
TRIANGLE_SIZE = 18.0        # データ座標系でのサイズ
TRIANGLE_ALPHA = 0.2
TRIANGLE_EVERY = 1          # 何点おきに△を描くか（1=全点）

REQUIRED_COLS = [
    "elapsed_time",
    "player_posx", "player_posy",
    "target_posx", "target_posy",
    "target_rotz",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_radians_if_needed(angle_series: pd.Series) -> np.ndarray:
    a = angle_series.to_numpy(dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return a
    max_abs = np.max(np.abs(finite))
    if max_abs > 2.0 * math.pi + 0.5:
        return np.deg2rad(a)
    return a


def triangle_points_sharp(x: float, y: float, theta_rad: float, size: float) -> np.ndarray:
    """
    頂角は鋭めだが細すぎない二等辺三角形（自前ポリゴン）
    tip が進行方向．
    """
    tip = np.array([size, 0.0])
    left = np.array([-size * 0.75,  size * 0.40])
    right = np.array([-size * 0.75, -size * 0.40])

    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array([[c, -s],
                  [s,  c]])

    pts = np.stack([tip, left, right], axis=0) @ R.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def heading_to_next(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.zeros(n)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    theta = np.arctan2(dy, dx)
    theta = np.append(theta, theta[-1])
    return theta


def plot_trajectory(df_full: pd.DataFrame, out_pdf_path: str, filename_base: str) -> None:
    df_full = df_full.sort_values("elapsed_time").reset_index(drop=True)

    # 全点を対象（必要なら ::10 などに変更）
    df = df_full.iloc[::1].reset_index(drop=True)

    # クリア秒数（小数第1位）
    clear_sec = float(df_full["elapsed_time"].iloc[-1] - df_full["elapsed_time"].iloc[0])

    # ファイル名形式: YYYYMMDD-HHmmss-Rx
    match = re.search(r"-R(\d+)$", filename_base)
    round_number = match.group(1) if match else "?"
    title_text = f"Round {round_number} (time: {clear_sec:.1f}s)"

    # 軌跡（全点）
    px_all = df_full["player_posx"].to_numpy(dtype=float)
    py_all = df_full["player_posy"].to_numpy(dtype=float)
    tx_all = df_full["target_posx"].to_numpy(dtype=float)
    ty_all = df_full["target_posy"].to_numpy(dtype=float)

    # △用
    px = df["player_posx"].to_numpy(dtype=float)
    py = df["player_posy"].to_numpy(dtype=float)
    ptheta = heading_to_next(px, py)

    # targetは固定：最初の位置＋向き（target_rotz代表値）
    trot = to_radians_if_needed(df_full["target_rotz"])
    ttheta = float(trot[np.isfinite(trot)][0]) if np.isfinite(trot).any() else 0.0
    tx0 = float(df_full["target_posx"].iloc[0])
    ty0 = float(df_full["target_posy"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    # 目盛（内側）
    ax.tick_params(axis="x", direction="in", pad=-12, top=True)
    ax.tick_params(axis="y", direction="in", pad=-14, left=True, right=True, labelleft=True)

    # X軸：200刻み（±1000除外）
    xticks = np.arange(-1000, 1001, 200)
    xticks = xticks[(xticks != -1000) & (xticks != 1000)]
    ax.set_xticks(xticks)

    # Y軸：200刻み（±600除外）
    yticks = np.arange(-600, 601, 200)
    yticks = yticks[(yticks != -600) & (yticks != 600)]
    ax.set_yticks(yticks)

    # 表示値を1/100スケールに
    formatter = FuncFormatter(lambda x, pos: f"{x/100:.0f}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 軌跡ライン
    ax.plot(px_all, py_all, color=PLAYER_COLOR, linewidth=LINE_WIDTH, alpha=0.85, zorder=3)
    ax.plot(tx_all, ty_all, color=TARGET_COLOR, linewidth=LINE_WIDTH, alpha=0.85, zorder=3)

    # player：鋭角△（自前ポリゴン）
    if len(df) >= 1:
        idx = np.arange(0, len(df), TRIANGLE_EVERY, dtype=int)
        for i in idx:
            if not (np.isfinite(px[i]) and np.isfinite(py[i]) and np.isfinite(ptheta[i])):
                continue
            tri = triangle_points_sharp(px[i], py[i], float(ptheta[i]), TRIANGLE_SIZE)
            ax.fill(
                tri[:, 0], tri[:, 1],
                color=PLAYER_COLOR,
                alpha=TRIANGLE_ALPHA,
                linewidth=0,
                zorder=6
            )

    # target：固定鋭角△（最前面）
    ttri = triangle_points_sharp(tx0, ty0, float(ttheta), TRIANGLE_SIZE)
    ax.fill(
        ttri[:, 0], ttri[:, 1],
        color=TARGET_COLOR,
        alpha=1.00,
        linewidth=0,
        zorder=20
    )

    # タイトル（内部）
    ax.text(
        0.02, 0.98, title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        zorder=10,
    )

    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.03, top=0.995)
    fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not csv_paths:
        print(f"csvが見つかりません: {INPUT_DIR}")
        return

    for csv_path in csv_paths:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_pdf = os.path.join(OUTPUT_DIR, f"{base}.pdf")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] 読み込み失敗: {csv_path} 例外: {e}")
            continue

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"[SKIP] 列不足: {csv_path} 不足列: {missing}")
            continue

        df_use = df[REQUIRED_COLS].copy()
        df_use = df_use.dropna(
            subset=["elapsed_time", "player_posx", "player_posy", "target_posx", "target_posy"],
            how="any",
        )

        if len(df_use) == 0:
            print(f"[SKIP] 有効行なし: {csv_path}")
            continue

        plot_trajectory(df_use, out_pdf, filename_base=base)
        print(f"[OK] {out_pdf}")


if __name__ == "__main__":
    main()
