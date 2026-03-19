import os

import numpy as np
import pyvista as pv
import trimesh
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    if mesh.vertices.shape[0] == 0:
        raise ValueError("无法读取到顶点，请检查文件。")
    return mesh


def ensure_right_handed(lr: np.ndarray, ap: np.ndarray, si: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.dot(np.cross(lr, ap), si) < 0:
        ap = -ap
    return lr, ap, si


def compute_symmetry_score(plane_params: np.ndarray, vertices: np.ndarray) -> float:
    normal = plane_params[:3]
    normal = normal / np.linalg.norm(normal)
    d = plane_params[3]
    distances = np.dot(vertices, normal) + d
    reflected_points = vertices - 2 * (distances.reshape(-1, 1) * normal)
    tree = cKDTree(reflected_points)
    distances_to_nearest, _ = tree.query(vertices)
    mean_distance = np.mean(distances_to_nearest)
    model_size = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    normalized_score = mean_distance / model_size
    return float(normalized_score)


def find_sagittal_by_symmetry(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    initial_normal = np.array([1.0, 0.0, 0.0], dtype=float)
    initial_guess = np.append(initial_normal, 0.0)
    bounds = [(-1, 1), (-1, 1), (-1, 1), (-10, 10)]
    result = minimize(
        compute_symmetry_score,
        initial_guess,
        args=(vertices_centered,),
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "maxiter": 500,
            "ftol": 1e-6,
            "gtol": 1e-6,
            "disp": False,
        },
    )
    optimal_params = result.x
    optimal_normal = optimal_params[:3]
    optimal_normal = optimal_normal / np.linalg.norm(optimal_normal)
    optimal_d = float(optimal_params[3])
    if optimal_normal[0] < 0:
        optimal_normal = -optimal_normal
        optimal_d = -optimal_d
    optimal_d = optimal_d - float(np.dot(optimal_normal, center))
    return center, optimal_normal


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, v_t = np.linalg.svd(points - centroid, full_matrices=False)
    normal = v_t[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal


def orient_axes_by_anterior(
    mesh: trimesh.Trimesh, origin: np.ndarray, lr: np.ndarray, ap: np.ndarray, si: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ap[1] < 0:
        ap = -ap
    if si[2] < 0:
        si = -si
        lr = -lr
    if lr[0] < 0:
        lr = -lr
    lr_u = lr - ap * (ap.dot(lr)) - si * (si.dot(lr))
    lr_u /= np.linalg.norm(lr_u)
    ap_u = ap - si * (si.dot(ap))
    ap_u /= np.linalg.norm(ap_u)
    si_u = si / np.linalg.norm(si)
    lr_u, ap_u, si_u = ensure_right_handed(lr_u, ap_u, si_u)
    return lr_u, ap_u, si_u


def build_transform_matrices(
    origin: np.ndarray, lr: np.ndarray, ap: np.ndarray, si: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r = np.column_stack([lr, ap, si])
    r_inv = r.T
    t = origin.reshape(3, 1)
    h_mesh_to_anat = np.eye(4)
    h_mesh_to_anat[:3, :3] = r_inv
    h_mesh_to_anat[:3, 3] = (-r_inv @ t).ravel()
    h_anat_to_mesh = np.eye(4)
    h_anat_to_mesh[:3, :3] = r
    h_anat_to_mesh[:3, 3] = t.ravel()
    return h_mesh_to_anat, h_anat_to_mesh


def _connected_components_from_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.size == 0:
        raise ValueError("Mesh 缺少可用边信息，无法计算连通域。")

    keep_edges = mask[edges[:, 0]] & mask[edges[:, 1]]
    edges = edges[keep_edges]
    if edges.shape[0] == 0:
        return []

    idx = np.flatnonzero(mask)
    idx_map = -np.ones(mask.shape[0], dtype=np.int64)
    idx_map[idx] = np.arange(idx.shape[0], dtype=np.int64)

    a = idx_map[edges[:, 0]]
    b = idx_map[edges[:, 1]]
    valid = (a >= 0) & (b >= 0)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return []

    adj: list[list[int]] = [[] for _ in range(idx.shape[0])]
    for i in range(a.size):
        ai = int(a[i])
        bi = int(b[i])
        adj[ai].append(bi)
        adj[bi].append(ai)

    visited = np.zeros(idx.shape[0], dtype=bool)
    comps: list[np.ndarray] = []
    for start in range(idx.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comp_global = idx[np.asarray(comp, dtype=np.int64)]
        comps.append(comp_global)

    comps.sort(key=lambda x: x.shape[0], reverse=True)
    return comps


def _graph_erode_mask(mesh: trimesh.Trimesh, mask: np.ndarray, iters: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if iters <= 0:
        return mask

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.size == 0:
        return mask
    a = edges[:, 0]
    b = edges[:, 1]

    for _ in range(int(iters)):
        keep = mask.copy()
        boundary_a = mask[a] & (~mask[b])
        boundary_b = mask[b] & (~mask[a])
        if np.any(boundary_a):
            keep[a[boundary_a]] = False
        if np.any(boundary_b):
            keep[b[boundary_b]] = False
        mask = keep
        if not np.any(mask):
            break
    return mask


def _graph_dilate_mask(mesh: trimesh.Trimesh, mask: np.ndarray, iters: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if iters <= 0:
        return mask

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.size == 0:
        return mask
    a = edges[:, 0]
    b = edges[:, 1]

    for _ in range(int(iters)):
        out = mask.copy()
        sel_a = mask[a]
        sel_b = mask[b]
        if np.any(sel_a):
            out[b[sel_a]] = True
        if np.any(sel_b):
            out[a[sel_b]] = True
        mask = out
    return mask


def _graph_open_mask(mesh: trimesh.Trimesh, mask: np.ndarray, iters: int = 1) -> np.ndarray:
    mask_e = _graph_erode_mask(mesh, mask, iters=iters)
    return _graph_dilate_mask(mesh, mask_e, iters=iters)


def _plane_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0], dtype=float) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = np.cross(n, a)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def _points_in_convex_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    points = np.asarray(points, dtype=float)
    n = poly.shape[0]
    if n < 3:
        return np.zeros(points.shape[0], dtype=bool)
    edges = np.roll(poly, -1, axis=0) - poly
    rel = points[:, None, :] - poly[None, :, :]
    cross_z = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    return np.all(cross_z >= 0.0, axis=1) | np.all(cross_z <= 0.0, axis=1)


def _fill_component_holes_on_plane(points: np.ndarray, shrink: float = 0.9) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 10:
        return points

    c0, n0 = fit_plane(points)
    e1, e2 = _plane_basis_from_normal(n0)
    p2 = np.column_stack([(points - c0) @ e1, (points - c0) @ e2])
    if p2.shape[0] < 10:
        return points

    hull = ConvexHull(p2)
    poly = p2[hull.vertices]
    poly_center = poly.mean(axis=0)
    poly = poly_center + float(shrink) * (poly - poly_center)

    tree = cKDTree(p2)
    d, _ = tree.query(p2, k=min(6, p2.shape[0]))
    step = float(np.median(d[:, -1]))
    if not np.isfinite(step) or step <= 0:
        step = float(np.max(p2.max(axis=0) - p2.min(axis=0)) / 80.0)
    step = max(step, 1e-4)

    min_xy = poly.min(axis=0)
    max_xy = poly.max(axis=0)
    max_grid_points = 200_000
    span_x = float(max_xy[0] - min_xy[0])
    span_y = float(max_xy[1] - min_xy[1])
    if span_x > 0 and span_y > 0:
        approx_n = int(np.ceil(span_x / step) + 1) * int(np.ceil(span_y / step) + 1)
        if approx_n > max_grid_points:
            scale = float(np.sqrt(approx_n / max_grid_points))
            step *= scale

    gx = np.arange(min_xy[0], max_xy[0] + step, step)
    gy = np.arange(min_xy[1], max_xy[1] + step, step)
    grid = np.array([(x, y) for x in gx for y in gy], dtype=float)
    inside = _points_in_convex_polygon(grid, poly)
    grid_in = grid[inside]
    if grid_in.shape[0] == 0:
        return points

    filled = c0[None, :] + grid_in[:, 0:1] * e1[None, :] + grid_in[:, 1:2] * e2[None, :]
    return np.vstack([points, filled])


def _plane_rms(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> float:
    d = (points - centroid) @ normal
    return float(np.sqrt(np.mean(d * d)))


def extract_endplates_by_normals(
    mesh: trimesh.Trimesh,
    lr_raw: np.ndarray,
    cand_angle_min_deg: float = 80.0,
    cand_angle_max_deg: float = 90.0,
    open_iters: int = 8,
    fill_shrink: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(mesh.vertices, dtype=float)
    lr_u = lr_raw / np.linalg.norm(lr_raw)

    n = np.asarray(mesh.vertex_normals, dtype=float)
    n_norm = np.linalg.norm(n, axis=1)
    n = n / np.clip(n_norm, 1e-12, None)[:, None]

    dot_lr = np.abs(n @ lr_u)
    ang = np.degrees(np.arccos(np.clip(dot_lr, 0.0, 1.0)))
    cand_mask = (ang >= float(cand_angle_min_deg)) & (ang <= float(cand_angle_max_deg))
    if int(open_iters) > 0:
        cand_mask = _graph_open_mask(mesh, cand_mask, iters=int(open_iters))
    comps = _connected_components_from_mask(mesh, cand_mask)
    if len(comps) < 2:
        raise ValueError("候选点连通域不足 2 个，请检查模型或调整角度范围。")

    comp_a = comps[0]
    comp_b = comps[1]
    pts_a = verts[comp_a]
    pts_b = verts[comp_b]

    pts_a_filled = _fill_component_holes_on_plane(pts_a, shrink=fill_shrink)
    pts_b_filled = _fill_component_holes_on_plane(pts_b, shrink=fill_shrink)

    top_c, top_n = fit_plane(pts_a_filled)
    bot_c, bot_n = fit_plane(pts_b_filled)

    si_dir = top_c - bot_c
    if np.linalg.norm(si_dir) < 1e-8:
        si_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    si_dir = si_dir / np.linalg.norm(si_dir)

    return pts_a_filled, pts_b_filled, top_c, top_n, bot_c, bot_n, si_dir


def build_by_endplate_normals(mesh: trimesh.Trimesh, visualize: bool = False, open_iters: int = 8, cand_angle_min_deg: float = 80.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(mesh.vertices, dtype=float)
    origin = verts.mean(axis=0)

    c_sym, lr_raw = find_sagittal_by_symmetry(verts)
    top_pts, bot_pts, top_c, top_n, bot_c, bot_n, si_dir = extract_endplates_by_normals(mesh, lr_raw, open_iters=open_iters, cand_angle_min_deg=cand_angle_min_deg)

    lr_u = lr_raw / np.linalg.norm(lr_raw)
    si_u = np.asarray(si_dir, dtype=float) - lr_u * float(np.dot(lr_u, si_dir))
    si_u = si_u / np.linalg.norm(si_u)
    if si_u[2] < 0:
        si_u = -si_u
    ap_u = np.cross(si_u, lr_u)
    ap_u = ap_u / np.linalg.norm(ap_u)
    lr_u, ap_u, si_u = ensure_right_handed(lr_u, ap_u, si_u)

    if visualize:
        pv_mesh = pv.PolyData(mesh.vertices, np.column_stack([np.full(len(mesh.faces), 3), mesh.faces]))
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color="lightblue", opacity=0.6, show_edges=True)
        bounds = pv_mesh.bounds
        size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        sagittal_plane = pv.Plane(center=c_sym, direction=lr_u, i_size=size * 1.5, j_size=size * 1.5)
        plotter.add_mesh(sagittal_plane, color="red", opacity=0.3, label="Sagittal Plane")
        top_plane = pv.Plane(center=top_c, direction=top_n, i_size=size * 0.7, j_size=size * 0.7)
        plotter.add_mesh(top_plane, color="green", opacity=0.4, label="Top Endplate")
        bot_plane = pv.Plane(center=bot_c, direction=bot_n, i_size=size * 0.7, j_size=size * 0.7)
        plotter.add_mesh(bot_plane, color="blue", opacity=0.4, label="Bottom Endplate")
        plotter.add_mesh(pv.PolyData(top_pts), color="green", point_size=3, render_points_as_spheres=True)
        plotter.add_mesh(pv.PolyData(bot_pts), color="blue", point_size=3, render_points_as_spheres=True)
        plotter.add_axes()
        plotter.add_legend()
        plotter.show()

    return origin, lr_u, ap_u, si_u


def main(input_file: str, output_path: str | None = None, visualize: bool = False, open_iters: int = 8, cand_angle_min_deg: float = 80.0) -> None:
    mesh = load_mesh(input_file)
    print(f"Loaded mesh: {input_file}  vertices={len(mesh.vertices)} faces={len(mesh.faces)}")

    origin, lr, ap, si = build_by_endplate_normals(mesh, visualize=visualize, open_iters=open_iters, cand_angle_min_deg=cand_angle_min_deg)
    lr, ap, si = orient_axes_by_anterior(mesh, origin, lr, ap, si)
    _, h_am = build_transform_matrices(origin, lr, ap, si)

    if output_path:
        output_file = output_path
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_coordinate.txt"

    np.savetxt(output_file, h_am, fmt="%.8f")
    print(f"Saved anat->mesh matrix to {output_file}")
    print("Origin (mesh coords):", origin)
    print("LR (mesh coords):", lr)
    print("AP (mesh coords):", ap)
    print("SI (mesh coords):", si)
    print("Done.")


def rerun_failed_list(failed_list_txt: str, output_dir: str, visualize: bool = False) -> None:
    if not os.path.exists(failed_list_txt):
        raise FileNotFoundError(f"失败文件列表不存在: {failed_list_txt}")

    os.makedirs(output_dir, exist_ok=True)
    with open(failed_list_txt, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    input_files = []
    seen = set()
    for line in raw_lines:
        if not line:
            continue
        path = line.split("\t", 1)[0].strip()
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        input_files.append(path)

    failed_again = []
    for input_file in input_files:
        base = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base.replace(".ply", "_coordinate.txt"))
        try:
            open_iters = 14
            main(input_file, output_file, visualize=visualize, open_iters=open_iters)
        except Exception as e:
            failed_again.append((input_file, repr(e)))
            print(f"重跑失败: {input_file}  error={e}")

    if failed_again:
        failed_again_txt = os.path.join(output_dir, "failed_again_coordinate_endplate_normals.txt")
        with open(failed_again_txt, "w", encoding="utf-8") as f:
            for path, err in failed_again:
                f.write(f"{path}\t{err}\n")
        print(f"已记录二次失败文件列表: {failed_again_txt}  count={len(failed_again)}")


if __name__ == "__main__":
    input_file = os.path.join("data", "example", "original_mesh.ply")
    output_dir = os.path.join("outputs", "coordinate_system")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "original_mesh_coordinate.txt")
    visualize = True
    open_iters = 6
    cand_angle_min_deg = 85
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Missing required file: {input_file}. Place your data under data/example/ or update the paths in __main__."
        )

    main(input_file, output_path, visualize, open_iters, cand_angle_min_deg)
