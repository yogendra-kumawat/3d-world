from __future__ import annotations
import struct
import io
from typing import Dict
import numpy as np
import cv2
from tactile_vision.types import PinMatrix

MODE_IDS: Dict[str, int] = {"walk": 1, "scene": 2, "identifier": 3}


def to_bytes(pm: PinMatrix, frame_index: int) -> bytes:
    """Serialize PinMatrix per the wire format pinned in the spec."""
    header = struct.pack(
        "<IBBBB",
        frame_index & 0xFFFFFFFF,
        pm.rows,
        pm.cols,
        pm.levels,
        MODE_IDS[pm.mode],
    )
    return header + pm.data.tobytes(order="C")


def render_pin_matrix_bgr(pm: PinMatrix, cell_px: int = 32) -> np.ndarray:
    """Fast OpenCV pin-matrix visualization. ~1ms per frame vs matplotlib's ~150ms.

    Layout: heatmap on top (cell color = pin height) + bar chart of column max
    along the bottom + frame text. Returns a BGR uint8 image.
    """
    rows, cols = pm.data.shape
    # Normalize pin levels to [0, 255] for colormap.
    norm = (pm.data.astype(np.float32) / max(pm.levels - 1, 1) * 255).astype(np.uint8)

    # Heatmap (rows*cell_px, cols*cell_px, 3).
    heatmap_small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    heatmap = cv2.resize(
        heatmap_small,
        (cols * cell_px, rows * cell_px),
        interpolation=cv2.INTER_NEAREST,
    )

    # Grid lines.
    for r in range(rows + 1):
        y = r * cell_px
        cv2.line(heatmap, (0, y), (cols * cell_px, y), (40, 40, 40), 1)
    for c in range(cols + 1):
        x = c * cell_px
        cv2.line(heatmap, (x, 0), (x, rows * cell_px), (40, 40, 40), 1)

    # Pin-height numbers in each cell.
    for r in range(rows):
        for c in range(cols):
            v = int(pm.data[r, c])
            if v > 0:
                cv2.putText(
                    heatmap,
                    str(v),
                    (c * cell_px + 6, r * cell_px + cell_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    # Bar chart strip below: max height per column.
    bar_h = 80
    bar_strip = np.zeros((bar_h, cols * cell_px, 3), dtype=np.uint8)
    col_max = pm.data.max(axis=0).astype(np.float32) / max(pm.levels - 1, 1)
    for c in range(cols):
        h_px = int(col_max[c] * (bar_h - 8))
        x0 = c * cell_px + 4
        x1 = (c + 1) * cell_px - 4
        cv2.rectangle(bar_strip, (x0, bar_h - 4), (x1, bar_h - 4 - h_px), (0, 200, 255), -1)

    # Header strip with title.
    header_h = 40
    header = np.zeros((header_h, cols * cell_px, 3), dtype=np.uint8)
    title = f"PinMatrix [{pm.mode}]  t={pm.timestamp:.2f}s  max={int(pm.data.max())}/{pm.levels - 1}  active={int((pm.data > 0).sum())}/{rows * cols}"
    cv2.putText(
        header,
        title,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([header, heatmap, bar_strip])


# Backwards-compat shim for the old name. Returns PNG bytes encoded by OpenCV.
def render_bar_chart_png(pm: PinMatrix) -> bytes:
    img = render_pin_matrix_bgr(pm)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def render_pin_matrix_matplotlib_3d(
    pm: PinMatrix,
    elev_deg: float = 30.0,
    azim_deg: float = -55.0,
    out_w: int = 800,
    out_h: int = 600,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Matplotlib 3D bar chart of the pin matrix. Slower (~80-150 ms) but
    visually richer. Returns BGR uint8 image. Optionally overlays per-cell
    labels (used by Identifier mode for class names)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rows, cols = pm.data.shape
    fig = plt.figure(figsize=(out_w / 100, out_h / 100), dpi=100, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    xs = xs.ravel()
    ys = ys.ravel()
    zs = np.zeros_like(xs, dtype=np.float32)
    dz = pm.data.ravel().astype(float) / max(pm.levels - 1, 1)

    # Color each bar by its height using the inferno colormap.
    cmap = plt.get_cmap("inferno")
    colors = cmap(dz)
    ax.bar3d(xs, ys, zs, 0.85, 0.85, np.maximum(dz, 0.01), color=colors, shade=True, edgecolor=(0.2, 0.2, 0.2, 0.6))

    if labels:
        for i, lbl in enumerate(labels):
            if not lbl:
                continue
            r = i // cols
            c = i % cols
            h = dz[i]
            if h > 0.05:
                ax.text(c + 0.4, r + 0.4, h + 0.05, lbl, color="white", fontsize=7, ha="center")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_zlim(0, 1)
    ax.set_xlabel("col", color="white", fontsize=8)
    ax.set_ylabel("row", color="white", fontsize=8)
    ax.set_zlabel("pin", color="white", fontsize=8)
    title = f"PinMatrix 3D [{pm.mode}]  t={pm.timestamp:.2f}s  max={int(pm.data.max())}/{pm.levels - 1}"
    ax.set_title(title, color="white", fontsize=10)
    ax.view_init(elev=elev_deg, azim=azim_deg)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.5)
        axis.set_tick_params(colors="white", labelsize=7)
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
