"""Point-cloud renders.

Several flavors:
  - render_birds_eye: fast 2D top-down PNG via OpenCV (always works, not interactive).
  - render_side_view: side-on slice (forward x up).
  - render_obstacle_map: image-space heatmap colored by height above floor.
  - render_depth_heatmap: classic depth heatmap with meter readout.
  - build_plotly_pointcloud: interactive 3D Plotly figure (drag to rotate).
"""
from __future__ import annotations
import numpy as np
import cv2


def render_birds_eye(
    points_xyz: np.ndarray,
    height_above_floor: np.ndarray | None,
    obstacle_mask: np.ndarray | None = None,
    region_depth_m: float | None = None,
    region_width_m: float | None = None,
    out_w: int = 480,
    out_h: int = 360,
    min_height_m: float = 0.3,
) -> np.ndarray:
    """Top-down BEV: x->image-x, z->image-y (z=0 at bottom = camera position).
    Color = height above floor; below `min_height_m` is dimmed.

    If region_*_m is None, auto-fits to the 95th-percentile range of valid points
    so the BEV is always populated regardless of intrinsics or scene scale.
    """
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    pts_all = points_xyz.reshape(-1, 3)
    if height_above_floor is not None:
        h_all = height_above_floor.ravel()
    else:
        h_all = np.full(pts_all.shape[0], 1.0, dtype=np.float32)

    finite = np.isfinite(pts_all).all(axis=1) & (pts_all[:, 2] > 0.1)
    if obstacle_mask is not None:
        finite &= obstacle_mask.ravel().astype(bool)
    pts = pts_all[finite]
    h = h_all[finite]
    if pts.shape[0] == 0:
        cv2.putText(img, "no valid points", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return img

    if region_depth_m is None:
        region_depth_m = float(np.percentile(pts[:, 2], 95))
        region_depth_m = max(region_depth_m, 1.0)
    if region_width_m is None:
        region_width_m = 2.0 * float(np.percentile(np.abs(pts[:, 0]), 95))
        region_width_m = max(region_width_m, 1.0)

    in_z = pts[:, 2] <= region_depth_m
    in_x = np.abs(pts[:, 0]) <= region_width_m / 2
    keep = in_z & in_x
    pts, h = pts[keep], h[keep]
    if pts.shape[0] == 0:
        cv2.putText(img, "no points after region clip", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return img

    # Map x in [-W/2, W/2] -> [0, out_w]; z in [0, D] -> [out_h-1, 0] so close = bottom.
    px = ((pts[:, 0] + region_width_m / 2) / region_width_m * out_w).astype(int)
    py = (out_h - 1 - (pts[:, 2] / region_depth_m * out_h)).astype(int)
    px = np.clip(px, 0, out_w - 1)
    py = np.clip(py, 0, out_h - 1)

    h_norm = np.clip(h / 2.0, 0, 1)
    floor = h < min_height_m
    color_indices = (h_norm * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(color_indices.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    cmap[floor] = (60, 60, 60)
    img[py, px] = cmap

    cv2.circle(img, (out_w // 2, out_h - 5), 5, (0, 255, 255), -1)
    n_rings = max(2, int(region_depth_m / max(region_depth_m / 5, 1)))
    for i in range(1, n_rings + 1):
        m = i * region_depth_m / n_rings
        y = int(out_h - 1 - (m / region_depth_m * out_h))
        cv2.line(img, (0, y), (out_w, y), (40, 40, 40), 1)
        cv2.putText(img, f"{m:.1f}m", (4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"BEV  W={region_width_m:.1f}m  D={region_depth_m:.1f}m  pts={pts.shape[0]}",
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return img


def render_depth_heatmap(depth: np.ndarray) -> np.ndarray:
    """Classic depth heatmap (close=red, far=blue) with meter readout."""
    d = depth.copy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        d_min = float(np.percentile(d[valid], 2))
        d_max = float(np.percentile(d[valid], 98))
    else:
        d_min, d_max = 0.0, 1.0
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    else:
        t = (d - d_min) / (d_max - d_min)
        t = np.clip(1.0 - t, 0.0, 1.0)
        d_norm = (t * 255).astype(np.uint8)
    img = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    label = f"depth: close={d_min:.1f}m  far={d_max:.1f}m  (red=close, blue=far)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_obstacle_map(
    frame_bgr: np.ndarray,
    height_above_floor: np.ndarray | None,
    depth_m: np.ndarray | None = None,
    min_height_m: float = 0.3,
) -> np.ndarray:
    """Overlay obstacles on the frame, colored by height above the floor.
    If the ground plane is missing or degenerate, falls back to coloring by
    inverse depth (closer = warmer)."""
    h_img, w_img = frame_bgr.shape[:2]
    base_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor((base_gray * 0.4).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    use_height = (
        height_above_floor is not None
        and np.isfinite(height_above_floor).any()
        and (np.nanmax(height_above_floor) - np.nanmin(height_above_floor)) > 0.5
    )

    if use_height:
        h = height_above_floor.astype(np.float32)
        h_clipped = np.clip(h / 2.5, 0, 1)
        h_norm = (h_clipped * 255).astype(np.uint8)
        cmap = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
        obstacle_mask = (h >= min_height_m) & np.isfinite(h)
        n_obs = int(obstacle_mask.sum())
        if n_obs < (h_img * w_img) * 0.01:
            # Almost nothing got tagged — fall through to depth fallback.
            use_height = False
        else:
            out = base.copy()
            out[obstacle_mask] = cv2.addWeighted(base, 0.3, cmap, 0.7, 0)[obstacle_mask]
            label = f"obstacles by height (blue=low, red=high)  floor=gray  obs={n_obs}px"
            cv2.rectangle(out, (0, 0), (w_img, 28), (0, 0, 0), -1)
            cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return out

    if depth_m is None:
        cv2.putText(base, "no ground plane and no depth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return base
    d = depth_m.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return base
    d_min = float(np.percentile(d[valid], 2))
    d_max = float(np.percentile(d[valid], 98))
    if d_max - d_min < 1e-6:
        return base
    t = np.clip(1.0 - (d - d_min) / (d_max - d_min), 0, 1)
    cmap = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = cv2.addWeighted(base, 0.4, cmap, 0.6, 0)
    label = f"FALLBACK: by depth (no usable ground plane).  close={d_min:.1f}m  far={d_max:.1f}m"
    cv2.rectangle(out, (0, 0), (w_img, 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_depth_mesh_3d(
    depth_m: np.ndarray,
    frame_bgr: np.ndarray,
    out_w: int = 640,
    out_h: int = 480,
    stride: int = 6,
    elev_deg: float = 25.0,
    azim_deg: float = -65.0,
) -> np.ndarray:
    """Matplotlib 3D scatter rendered offscreen as a PNG. Slower (~80ms) but useful
    when you want a static 3D angle of the depth as a mesh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h_img, w_img = depth_m.shape
    yy, xx = np.mgrid[0:h_img, 0:w_img]
    sub_d = depth_m[::stride, ::stride]
    sub_x = xx[::stride, ::stride]
    sub_y = yy[::stride, ::stride]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    sub_rgb = rgb[::stride, ::stride].reshape(-1, 3) / 255.0

    valid = np.isfinite(sub_d.ravel()) & (sub_d.ravel() > 0)
    pts_x = sub_x.ravel()[valid]
    pts_y = sub_y.ravel()[valid]
    pts_d = sub_d.ravel()[valid]
    pts_rgb = sub_rgb[valid]

    fig = plt.figure(figsize=(out_w / 100, out_h / 100), dpi=100, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.scatter(pts_x, pts_d, -pts_y, c=pts_rgb, s=1, marker=".", depthshade=False)
    ax.view_init(elev=elev_deg, azim=azim_deg)
    ax.set_xlabel("x px", color="white", fontsize=8)
    ax.set_ylabel("depth m", color="white", fontsize=8)
    ax.set_zlabel("-y px (up)", color="white", fontsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.5)
        axis.set_tick_params(colors="white", labelsize=7)
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def render_depth_contours(depth_m: np.ndarray, n_levels: int = 12) -> np.ndarray:
    """Depth as topographic contour lines (close=red, far=blue) over a dim grayscale."""
    d = depth_m.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros(depth_m.shape + (3,), dtype=np.uint8)
    d_min = float(np.percentile(d[valid], 2))
    d_max = float(np.percentile(d[valid], 98))
    t = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    canvas = (t * 80).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    levels = np.linspace(0.05, 0.95, n_levels)
    for i, lv in enumerate(levels):
        mask = (np.abs(t - lv) < (0.5 / n_levels)).astype(np.uint8) * 255
        color = cv2.applyColorMap(np.array([[int((1 - lv) * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        canvas[mask > 0] = (int(color[0]), int(color[1]), int(color[2]))
    label = f"contours: close={d_min:.1f}m  far={d_max:.1f}m  ({n_levels} levels)"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_side_view(
    points_xyz: np.ndarray,
    height_above_floor: np.ndarray | None,
    out_w: int = 480,
    out_h: int = 360,
    min_height_m: float = 0.3,
) -> np.ndarray:
    """Side-view: forward (z) on x-axis, height (up) on y-axis.
    Camera at left edge; floor along the bottom; obstacles rise up."""
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    pts = points_xyz.reshape(-1, 3)
    if height_above_floor is not None:
        up = height_above_floor.ravel()
    else:
        up = -pts[:, 1]  # OpenCV: -y = up
    finite = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.1) & np.isfinite(up)
    pts = pts[finite]
    up = up[finite]
    if pts.shape[0] == 0:
        cv2.putText(img, "no valid points", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return img
    z_max = float(np.percentile(pts[:, 2], 95))
    z_max = max(z_max, 1.0)
    h_max = max(float(np.percentile(up, 99)), 1.0)
    h_min = min(float(np.percentile(up, 1)), -0.5)
    px = (pts[:, 2] / z_max * (out_w - 1)).astype(int)
    py = (out_h - 1 - ((up - h_min) / (h_max - h_min) * (out_h - 1))).astype(int)
    px = np.clip(px, 0, out_w - 1)
    py = np.clip(py, 0, out_h - 1)
    floor = up < min_height_m
    color_indices = (np.clip(up / 2.0, 0, 1) * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(color_indices.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    cmap[floor] = (60, 60, 60)
    img[py, px] = cmap
    # Floor reference line at h=0.
    floor_y = int(out_h - 1 - ((0 - h_min) / (h_max - h_min) * (out_h - 1)))
    cv2.line(img, (0, floor_y), (out_w, floor_y), (90, 90, 90), 1)
    cv2.putText(img, "floor", (4, floor_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.circle(img, (5, floor_y), 5, (0, 255, 255), -1)
    cv2.putText(img, f"side view  fwd<={z_max:.1f}m  up:[{h_min:.1f},{h_max:.1f}]m", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def render_view(
    view_name: str,
    *,
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    points_xyz: np.ndarray,
    height_above_floor: np.ndarray | None,
) -> np.ndarray:
    """Dispatch a view by name. Returns BGR image."""
    if view_name == "Depth heatmap":
        return render_depth_heatmap(depth)
    if view_name == "Top-down BEV":
        return render_birds_eye(points_xyz, height_above_floor)
    if view_name == "Side view":
        return render_side_view(points_xyz, height_above_floor)
    if view_name == "Obstacle map":
        return render_obstacle_map(frame_bgr, height_above_floor, depth_m=depth)
    if view_name == "Depth contours":
        return render_depth_contours(depth)
    if view_name == "3D depth mesh (matplotlib)":
        return render_depth_mesh_3d(depth, frame_bgr)
    return render_depth_heatmap(depth)


def build_plotly_pointcloud(
    points_xyz: np.ndarray,
    height_above_floor: np.ndarray | None,
    frame_bgr: np.ndarray,
    stride: int = 4,
    min_height_m: float = 0.3,
    max_depth_m: float = 30.0,
):
    """Interactive 3D point cloud as a Plotly figure (drag to rotate).
    Uses a regular grid stride so the cloud is structured, not noisy random samples.
    """
    import plotly.graph_objects as go

    h_img, w_img = points_xyz.shape[:2]
    sub_pts = points_xyz[::stride, ::stride].reshape(-1, 3)
    rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    sub_rgb = rgb_full[::stride, ::stride].reshape(-1, 3)
    if height_above_floor is not None:
        sub_h = height_above_floor[::stride, ::stride].ravel()
    else:
        sub_h = np.full(sub_pts.shape[0], 1.0, dtype=np.float32)

    finite = (
        np.isfinite(sub_pts).all(axis=1)
        & (sub_pts[:, 2] > 0.1)
        & (sub_pts[:, 2] < max_depth_m)
    )
    pts = sub_pts[finite]
    rgb = sub_rgb[finite]
    h_arr = sub_h[finite]
    if pts.shape[0] == 0:
        return go.Figure().update_layout(title="no valid points", paper_bgcolor="rgb(15,15,15)")

    floor = h_arr < min_height_m
    color_arr = rgb.copy().astype(np.int16)
    color_arr[floor] = color_arr[floor] // 3
    colors = [f"rgb({r},{g},{b})" for r, g, b in color_arr]

    # Plotly axes: x = left/right, y = forward (depth), z = up.
    x = pts[:, 0]
    y = pts[:, 2]
    z = -pts[:, 1]

    cloud_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=2, color=colors, opacity=0.9),
        hovertemplate="x=%{x:.2f}m<br>fwd=%{y:.2f}m<br>up=%{z:.2f}m<extra></extra>",
        name="points",
    )

    # Origin / camera marker.
    cam_trace = go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=8, color="yellow", symbol="diamond"),
        text=["camera"], textposition="top center",
        textfont=dict(color="yellow", size=12),
        name="camera",
    )

    # XYZ axis arrows from the origin (1m each).
    axes = []
    for vec, color, name in [
        ((1, 0, 0), "red", "+x right"),
        ((0, 1, 0), "lime", "+y forward"),
        ((0, 0, 1), "deepskyblue", "+z up"),
    ]:
        axes.append(
            go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                mode="lines+text",
                line=dict(color=color, width=6),
                text=["", name], textposition="top center",
                textfont=dict(color=color, size=10),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=[cloud_trace, cam_trace, *axes])

    # Auto-fit the bounding box.
    margin = 1.0
    x_range = [float(np.percentile(x, 1)) - margin, float(np.percentile(x, 99)) + margin]
    y_range = [0, float(np.percentile(y, 99)) + margin]
    z_range = [float(np.percentile(z, 1)) - margin, float(np.percentile(z, 99)) + margin]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m, left/right)", range=x_range, gridcolor="rgb(60,60,60)", color="rgb(220,220,220)", backgroundcolor="rgb(15,15,15)"),
            yaxis=dict(title="forward (m)",       range=y_range, gridcolor="rgb(60,60,60)", color="rgb(220,220,220)", backgroundcolor="rgb(15,15,15)"),
            zaxis=dict(title="up (m)",            range=z_range, gridcolor="rgb(60,60,60)", color="rgb(220,220,220)", backgroundcolor="rgb(15,15,15)"),
            aspectmode="data",
            bgcolor="rgb(15,15,15)",
            camera=dict(eye=dict(x=1.5, y=-2.5, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="rgb(15,15,15)",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        uirevision="static",
    )
    return fig
