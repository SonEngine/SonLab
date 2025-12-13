import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def build_point_cloud(points: np.ndarray, colors: np.ndarray | None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        if colors.size > 0 and colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def load_ply_geometry(ply_path: Path):
    """
    Returns (geometry)
    """
    # 1) Try point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.is_empty():

        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        pcd = build_point_cloud(pts, cols)
        return pcd

    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=str, required=True, help="Path to a .ply file.")
    args = ap.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        print(f"[ERROR] PLY not found: {ply_path}")
        return 1

    geom = load_ply_geometry(ply_path)

    o3d.visualization.draw_geometries(
        [geom],
        window_name=f"PLY Viewer - {ply_path.name}",
        width=1280,
        height=720,
        point_show_normal=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
