import os
import time
import numpy as np
import pyvista as pv

from generate_meshes import (
    generate_meshes,
    load_point_cloud,
    build_surface_graph_from_original_mesh,
    find_shortest_path,
    create_path_region,
    create_key_point_circle,
)
from post_process_mesh import post_process_meshes


def convert_ply_to_txt_for_generate_meshes(ply_path, original_mesh_path=None, txt_path=None):
    mesh = pv.read(ply_path)
    points = np.asarray(mesh.points)

    normals = None
    if original_mesh_path is not None and os.path.isfile(original_mesh_path):
        orig_mesh = pv.read(original_mesh_path)
        if "Normals" not in orig_mesh.point_data:
            orig_mesh.compute_normals(inplace=True)
        if "Normals" in orig_mesh.point_data:
            normals = np.asarray(orig_mesh.point_data["Normals"])

    if normals is None or normals.shape[0] != points.shape[0]:
        normals = np.zeros_like(points)

    n_points = points.shape[0]
    labels = np.zeros((n_points, 1), dtype=float)

    r = None
    g = None
    b = None
    if all(name in mesh.point_data for name in ["red", "green", "blue"]):
        r = np.asarray(mesh.point_data["red"]).reshape(-1)
        g = np.asarray(mesh.point_data["green"]).reshape(-1)
        b = np.asarray(mesh.point_data["blue"]).reshape(-1)
    elif "RGB" in mesh.point_data:
        rgb = np.asarray(mesh.point_data["RGB"])
        if rgb.ndim == 2 and rgb.shape[1] >= 3:
            r = rgb[:, 0]
            g = rgb[:, 1]
            b = rgb[:, 2]

    if r is not None and g is not None and b is not None:
        is_blue = (b > 200) & (r < 80) & (g < 80)
        is_red = (r > 200) & (g < 80) & (b < 80)
        labels[is_blue, 0] = 1.0
        labels[is_red, 0] = 2.0

    if txt_path is None:
        base, _ = os.path.splitext(ply_path)
        txt_path = base + ".txt"
    else:
        if os.path.isdir(txt_path):
            base_name = os.path.splitext(os.path.basename(ply_path))[0]
            txt_path = os.path.join(txt_path, base_name + ".txt")

    data = np.hstack([points, normals, labels])
    with open(txt_path, "w", encoding="utf-8") as f:
        for x, y, z, nx, ny, nz, l in data:
            f.write(f"{x} {y} {z} {nx} {ny} {nz} {int(l)}\n")

    return txt_path


def run_full_pipeline(input_file, original_mesh_file, json_file, stage1_output_dir, final_output_dir,
                      path_width, extrude_height, key_point_radius,
                      cylinder_radius, cylinder_height, cuboid_height):
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    left_mesh_file, right_mesh_file, _ = generate_meshes(
        input_file=input_file,
        original_mesh_file=original_mesh_file,
        output_dir=stage1_output_dir,
        path_width=path_width,
        extrude_height=extrude_height,
        key_point_radius=key_point_radius,
    )

    _, _, _, left_region_points, right_region_points, _ = load_point_cloud(
        input_file, original_mesh_file
    )

    output_file = post_process_meshes(
        left_mesh_file=left_mesh_file,
        right_mesh_file=right_mesh_file,
        original_mesh_file=original_mesh_file,
        left_region_points=left_region_points,
        right_region_points=right_region_points,
        json_file=json_file,
        output_dir=final_output_dir,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
        cuboid_height=cuboid_height,
    )

    return output_file


STYLE_CONFIG = {
    "background": "white",
    "font_color": "black",
    "mesh_base_color": "#7e8081",
    "key_point_common": "#00CC99",
    "region_points": "#E63900",
    "left_color": "#3399FF",
    "right_color": "#FF9933",
    "path_color": "#FFD700",
    "final_mesh_color": "#e3e0ca",
}

def visualize_pipeline_steps(ply_input_file, txt_file, original_mesh_file, json_file,
                             stage1_output_dir, final_output_file,
                             path_width, key_point_radius, screenshot_save_dir):
    
    original_mesh = pv.read(original_mesh_file)
    if "Normals" not in original_mesh.point_data:
        original_mesh.compute_normals(inplace=True, auto_orient_normals=True)

    pointcloud, left_key_point, right_key_point, left_region_points, right_region_points, _ = load_point_cloud(
        txt_file, original_mesh_file
    )
    
    graph = build_surface_graph_from_original_mesh(original_mesh, pointcloud)
    path_left = find_shortest_path(graph, pointcloud.points, left_key_point, left_region_points)
    path_right = find_shortest_path(graph, pointcloud.points, right_key_point, right_region_points)
    
    path_region_indices_left = create_path_region(pointcloud.points, path_left, left_region_points, path_width)
    path_region_points_left = pointcloud.points[path_region_indices_left]
    left_circle_indices = create_key_point_circle(original_mesh, left_key_point, key_point_radius)
    circle_points_left = original_mesh.points[left_circle_indices]
    base_region_left = np.unique(np.vstack((path_region_points_left, circle_points_left)), axis=0)
    
    path_region_indices_right = create_path_region(pointcloud.points, path_right, right_region_points, path_width)
    path_region_points_right = pointcloud.points[path_region_indices_right]
    right_circle_indices = create_key_point_circle(original_mesh, right_key_point, key_point_radius)
    circle_points_right = original_mesh.points[right_circle_indices]
    base_region_right = np.unique(np.vstack((path_region_points_right, circle_points_right)), axis=0)
    
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    left_mesh = pv.read(os.path.join(stage1_output_dir, f"{base_name}_left_mesh.stl"))
    right_mesh = pv.read(os.path.join(stage1_output_dir, f"{base_name}_right_mesh.stl"))
    final_mesh = pv.read(final_output_file)

    for m in [left_mesh, right_mesh, final_mesh]:
        if "Normals" not in m.point_data:
            m.compute_normals(inplace=True, auto_orient_normals=True)

    labels = pointcloud.point_data.get("Labels", None)
    key_glyphs = None
    if labels is not None:
        pts_all = pointcloud.points
        mask_key_all = labels == 1
        if np.any(mask_key_all):
            key_points_poly = pv.PolyData(pts_all[mask_key_all])
            key_glyphs = key_points_poly.glyph(scale=False, geom=pv.Sphere(radius=0.8), orient=False)

    plotter = pv.Plotter(lighting="none") 
    plotter.set_background(STYLE_CONFIG["background"])
    
    plotter.enable_ssao(radius=10, bias=0.5)
    
    plotter.enable_anti_aliasing("ssaa")
    
    light1 = pv.Light(position=(100, 100, 100), intensity=1.1, color='white') 
    light2 = pv.Light(position=(-100, 50, 50), intensity=0.7, color='#fffae0') 
    light3 = pv.Light(position=(0, 100, -100), intensity=0.5, color='white') 

    state = {"step": 0, "camera_initialized": False}
    max_step = 5

    def add_base_mesh(opacity=0.3):
        plotter.add_mesh(
            original_mesh,
            color=STYLE_CONFIG["mesh_base_color"],
            opacity=opacity,
            smooth_shading=True,
            specular=0.2,
            specular_power=15,
            diffuse=0.8,
            ambient=0.35,
        )

    def add_region_points():
        if labels is not None:
            pts = pointcloud.points
            mask_region = labels == 2
            plotter.add_mesh(
                pv.PolyData(pts[mask_region]),
                color=STYLE_CONFIG["region_points"],
                point_size=16,
                render_points_as_spheres=True,
                ambient=0.5,
                diffuse=0.8
            )
            if key_glyphs is not None:
                plotter.add_mesh(key_glyphs, color="black", opacity=0.6)

    def show_step(i):
        plotter.clear()
        plotter.remove_all_lights()
        plotter.add_light(light1)
        plotter.add_light(light2)
        plotter.add_light(light3)

        if i == 0:
            plotter.add_mesh(
                original_mesh,
                color=STYLE_CONFIG["mesh_base_color"],
                opacity=0.7,
                smooth_shading=True,
                specular=0.2,
                specular_power=15,
                diffuse=0.9,
                ambient=0.3
            )

        elif i == 1:
            add_base_mesh(opacity=0.35)
            add_region_points()
            
            plotter.add_mesh(pv.Sphere(center=left_key_point.flatten(), radius=1.2), 
                             color=STYLE_CONFIG["key_point_common"], smooth_shading=True,
                             ambient=0.3, diffuse=0.8)
            plotter.add_mesh(pv.Sphere(center=right_key_point.flatten(), radius=1.2), 
                             color=STYLE_CONFIG["key_point_common"], smooth_shading=True,
                             ambient=0.3, diffuse=0.8)

        elif i == 2:
            add_base_mesh(opacity=0.35)
            add_region_points()
            
            pts = pointcloud.points
            tube_left = pv.Spline(pts[path_left], 100).tube(radius=0.2)
            tube_right = pv.Spline(pts[path_right], 100).tube(radius=0.2)
            
            plotter.add_mesh(tube_left, color=STYLE_CONFIG["path_color"], smooth_shading=True, ambient=1.0)
            plotter.add_mesh(tube_right, color=STYLE_CONFIG["path_color"], smooth_shading=True, ambient=1.0)
            
            plotter.add_mesh(pv.Sphere(center=left_key_point.flatten(), radius=1.0), 
                             color=STYLE_CONFIG["key_point_common"], ambient=0.3)
            plotter.add_mesh(pv.Sphere(center=right_key_point.flatten(), radius=1.0), 
                             color=STYLE_CONFIG["key_point_common"], ambient=0.3)

        elif i == 3:
            add_base_mesh(opacity=0.35)
            plotter.add_mesh(pv.PolyData(base_region_left), color=STYLE_CONFIG["left_color"], 
                             point_size=6, render_points_as_spheres=True, ambient=0.7)
            plotter.add_mesh(pv.PolyData(base_region_right), color=STYLE_CONFIG["right_color"], 
                             point_size=6, render_points_as_spheres=True, ambient=0.7)

        elif i == 4:
            add_base_mesh(opacity=0.35)
            plotter.add_mesh(left_mesh, color=STYLE_CONFIG["left_color"], opacity=1.0, 
                             smooth_shading=True, specular=0.3, diffuse=0.8, ambient=0.3)
            plotter.add_mesh(right_mesh, color=STYLE_CONFIG["right_color"], opacity=1.0, 
                             smooth_shading=True, specular=0.3, diffuse=0.8, ambient=0.3)

        elif i == 5:
            add_base_mesh(opacity=0.35)
            
            plotter.add_mesh(
                final_mesh,
                color=STYLE_CONFIG["final_mesh_color"],
                opacity=1.0,
                smooth_shading=True,
                specular=0.6,        
                specular_power=40,   
                diffuse=0.9,
                ambient=0.25,
            )

        if not state["camera_initialized"]:
            plotter.reset_camera()
            state["camera_initialized"] = True

        plotter.render()

    def forward():
        if state["step"] < max_step:
            state["step"] += 1
            show_step(state["step"])

    def backward():
        if state["step"] > 0:
            state["step"] -= 1
            show_step(state["step"])
            
    def save_shot():
        if not os.path.exists(screenshot_save_dir):
            os.makedirs(screenshot_save_dir)
        filename = os.path.join(screenshot_save_dir, f"{time.strftime('%Y%m%d%H%M%S')}.png")
        plotter.screenshot(filename, transparent_background=True, window_size=[2048, 2048])
        print(f"Snapshot saved: {filename}")

    plotter.add_key_event("n", lambda: forward())
    plotter.add_key_event("p", lambda: backward())
    plotter.add_key_event("s", lambda: save_shot())
    
    show_step(0)
    plotter.show()


if __name__ == "__main__":
    ply_input_file = os.path.join("data", "example", "labeled_colored.ply")
    original_mesh_file = os.path.join("data", "example", "original_mesh.ply")
    json_file = os.path.join("data", "example", "trajectory.json")

    output_root = os.path.join("outputs", "full_pipeline")
    txt_dir = os.path.join(output_root, "stage0_txt")
    stage1_output_dir = os.path.join(output_root, "stage1_meshes")
    final_output_dir = os.path.join(output_root, "stage2_guide_plate")
    screenshot_save_dir = os.path.join(output_root, "screenshots")

    path_width = 3.5
    extrude_height = 2.0
    key_point_radius = 6.0

    cylinder_radius = 0.75
    cylinder_height = 20.0
    cuboid_height = 1.0

    for p in [ply_input_file, original_mesh_file, json_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required file: {p}. Place your data under data/example/ or update the paths in __main__."
            )

    input_file = convert_ply_to_txt_for_generate_meshes(ply_input_file, original_mesh_file, txt_dir)

    output_file = run_full_pipeline(
        input_file=input_file,
        original_mesh_file=original_mesh_file,
        json_file=json_file,
        stage1_output_dir=stage1_output_dir,
        final_output_dir=final_output_dir,
        path_width=path_width,
        extrude_height=extrude_height,
        key_point_radius=key_point_radius,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
        cuboid_height=cuboid_height,
    )

    visualize_pipeline_steps(
        ply_input_file=ply_input_file,
        txt_file=input_file,
        original_mesh_file=original_mesh_file,
        json_file=json_file,
        stage1_output_dir=stage1_output_dir,
        final_output_file=output_file,
        path_width=path_width,
        key_point_radius=key_point_radius,
        screenshot_save_dir=screenshot_save_dir,
    )
