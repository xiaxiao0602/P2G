import os
import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial import cKDTree


def load_point_cloud(input_file, original_mesh_file=None):
    """
    Load a point cloud and split key points / region points into left and right.

    Parameters
    ----------
    input_file : str
        TXT point cloud file. Each line is: x y z nx ny nz label
        label=1 for key points, label=2 for region points.
    original_mesh_file : str | None
        Optional original mesh file (e.g., .ply) for downstream surface path computation.

    Returns
    -------
    tuple
        (pointcloud, left_key_point, right_key_point, left_region_points, right_region_points, original_mesh)
    """
    print(f"Loading point cloud: {input_file}")

    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.txt':
        print("Detected TXT point cloud file")
        points = []
        normals = []
        labels = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 7:
                    x, y, z, nx, ny, nz, label = map(float, values[:7])
                    points.append([x, y, z])
                    normals.append([nx, ny, nz])
                    labels.append(label)
        
        points = np.array(points)
        normals = np.array(normals)
        labels = np.array(labels)
        
        pointcloud = pv.PolyData(points)
        pointcloud.point_data['Normals'] = normals
        pointcloud.point_data['Labels'] = labels
        
        key_points = points[labels == 1]
        region_points = points[labels == 2]
        
        print(f"Loaded {len(points)} points from TXT")
        print(f"Found {len(key_points)} key points and {len(region_points)} region points")
    else:
        raise ValueError(f"Unsupported input format: {file_ext}. Expected a .txt point cloud file.")
    
    original_mesh = None
    if original_mesh_file and os.path.exists(original_mesh_file):
        print(f"Loading original mesh: {original_mesh_file}")
        original_mesh = pv.read(original_mesh_file)
    
    if len(key_points) > 0 and len(region_points) > 0:
        x_center = np.median(points[:, 0])
        
        left_key_points = key_points[key_points[:, 0] < x_center]
        right_key_points = key_points[key_points[:, 0] >= x_center]
        
        left_region_points = region_points[region_points[:, 0] < x_center]
        right_region_points = region_points[region_points[:, 0] >= x_center]
        
        print(f"Left: {len(left_key_points)} key points, {len(left_region_points)} region points")
        print(f"Right: {len(right_key_points)} key points, {len(right_region_points)} region points")
        
        if len(left_key_points) > 1:
            left_key_point = np.mean(left_key_points, axis=0).reshape(1, -1)
            print(f"Warning: multiple left key points found, using centroid: {left_key_point}")
        else:
            left_key_point = left_key_points[0].reshape(1, -1)
        
        if len(right_key_points) > 1:
            right_key_point = np.mean(right_key_points, axis=0).reshape(1, -1)
            print(f"Warning: multiple right key points found, using centroid: {right_key_point}")
        else:
            right_key_point = right_key_points[0].reshape(1, -1)
        
        return pointcloud, left_key_point, right_key_point, left_region_points, right_region_points, original_mesh
    else:
        raise ValueError("Insufficient key points or region points (need labels 1 and 2).")

def build_surface_graph_from_original_mesh(original_mesh, pointcloud):
    """
    Build a surface graph from the original mesh to compute geodesic shortest paths.

    Parameters
    ----------
    original_mesh : pv.PolyData
        Original mesh with faces.
    pointcloud : pv.PolyData
        Target point cloud (a subset/resampling of the original mesh vertices).

    Returns
    -------
    nx.Graph
        Graph whose nodes correspond to pointcloud point indices.
    """
    print("Building surface graph from original mesh...")
    
    original_tree = cKDTree(original_mesh.points)
    
    target_points = pointcloud.points
    
    _, target_to_original = original_tree.query(target_points)
    
    orig_idx, first_target = np.unique(target_to_original, return_index=True)
    original_to_target = dict(zip(orig_idx.tolist(), first_target.tolist()))
    
    G = nx.Graph()
    
    for i in range(len(target_points)):
        G.add_node(i)
    
    print(f"Original mesh has {original_mesh.n_cells} faces")
    
    edges_added = 0
    for i in range(original_mesh.n_cells):
        cell = original_mesh.get_cell(i)
        if cell.type == 5:  # VTK_TRIANGLE
            face_vertices = cell.point_ids
            
            target_vertices = []
            for v in face_vertices:
                if v in original_to_target:
                    target_vertices.append(original_to_target[v])
            
            if len(target_vertices) >= 2:
                for j in range(len(target_vertices)):
                    for k in range(j + 1, len(target_vertices)):
                        v1, v2 = target_vertices[j], target_vertices[k]
                        
                        weight = np.linalg.norm(target_points[v1] - target_points[v2])
                        
                        if G.has_edge(v1, v2):
                            current_weight = G[v1][v2]['weight']
                            if weight < current_weight:
                                G[v1][v2]['weight'] = weight
                        else:
                            G.add_edge(v1, v2, weight=weight)
                            edges_added += 1
    
    print(f"Surface graph built, added {edges_added} edges")
    
    return G

def find_shortest_path(G, points, start_point, target_points):
    """
    Find a shortest path from a start point to the boundary of a target region.

    Parameters
    ----------
    G : nx.Graph
        Surface graph.
    points : np.ndarray
        Point coordinates (N, 3).
    start_point : np.ndarray
        Start point coordinate (shape (1,3) or (3,)).
    target_points : np.ndarray
        Region point coordinates (M, 3).

    Returns
    -------
    list[int]
        Indices along the shortest path.
    """
    print("Searching shortest path...")
    
    tree = cKDTree(points)
    _, start_idx = tree.query(start_point.flatten())
    
    _, region_indices = tree.query(target_points)
    region_indices = list(set(region_indices))  
    
    boundary_indices = list(nx.node_boundary(G, region_indices))
    
    print(f"Found {len(boundary_indices)} boundary nodes")
    
    shortest_path = None
    min_distance = float('inf')
    
    for target_idx in boundary_indices:
        try:
            path = nx.shortest_path(G, source=start_idx, target=target_idx, weight='weight')
            distance = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            
            if distance < min_distance:
                min_distance = distance
                shortest_path = path
        except nx.NetworkXNoPath:
            continue
    
    if shortest_path is None:
        raise ValueError("No path found from the key point to the region boundary")
    
    print(f"Shortest path length: {min_distance}, nodes: {len(shortest_path)}")
    return shortest_path

def create_path_region(points, path, region_points, path_width):
    """
    Create an expanded region around a path and merge with the original region.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates (N, 3).
    path : list[int]
        Path indices.
    region_points : np.ndarray
        Region point coordinates (M, 3).
    path_width : float
        Radius used to expand the path.

    Returns
    -------
    list[int]
        Indices of points in the merged region.
    """
    print("Creating path region...")
    path_points = np.array([points[idx] for idx in path])

    tree = cKDTree(points)
    path_region_indices = set()
    for point in path_points:
        indices = tree.query_ball_point(point, r=path_width)
        path_region_indices.update(indices)
    
    region_indices = set()
    for point in region_points:
        _, idx = tree.query(point)
        region_indices.add(idx)
    
    combined_region = path_region_indices.union(region_indices)
    print(f"Region points in merged set: {len(combined_region)}")
    
    valid_indices = [idx for idx in combined_region if idx < len(points)]
    
    return valid_indices

def extrude_region(extrude_distance=2.0, original_mesh=None, region_points=None):
    """
    Extrude a region outward using the original mesh normals and reconstruct a surface.

    Parameters
    ----------
    extrude_distance : float
        Extrusion distance in the same unit as the mesh.
    original_mesh : pv.PolyData
        Reference mesh used for normal lookup.
    region_points : np.ndarray
        Region point coordinates (M, 3).

    Returns
    -------
    pv.PolyData
        Reconstructed surface mesh of the extruded region.
    """
    print("Extruding region...")
    if 'Normals' not in original_mesh.point_data:
        original_mesh.compute_normals(inplace=True)
    
    tree = cKDTree(original_mesh.points)
    
    closest_points_indices = []
    for point in region_points:
        _, idx = tree.query(point)
        closest_points_indices.append(idx)
    
    normals = -original_mesh.point_data['Normals'][closest_points_indices]
    
    extruded_points = region_points + normals * extrude_distance
    
    points = np.vstack((region_points, extruded_points))
    normals = np.vstack((normals, -normals))
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    print("Running Poisson surface reconstruction...")
    mesh_rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, scale=1.5)

    o3d_vertices = np.asarray(mesh_rec.vertices)
    o3d_triangles = np.asarray(mesh_rec.triangles)
    faces = []
    for triangle in o3d_triangles:
        faces.append(3)
        faces.extend(triangle)
    extruded_mesh = pv.PolyData(o3d_vertices, faces)

    extruded_mesh = extruded_mesh.fill_holes(500.0) 
    
    return extruded_mesh

def create_key_point_circle(original_mesh, key_point, radius=5.0):
    """
    Create a circular neighborhood on the original mesh around a key point.

    Parameters
    ----------
    original_mesh : pv.PolyData
        Original mesh.
    key_point : np.ndarray
        Key point coordinate (shape (1,3) or (3,)).
    radius : float
        Radius on the mesh (Euclidean distance on vertices).

    Returns
    -------
    list[int]
        Vertex indices in the neighborhood.
    """
    print(f"Creating key-point neighborhood, radius={radius} ...")
    
    if original_mesh is None or original_mesh.n_points == 0:
        raise ValueError("Invalid original_mesh: empty mesh")
    
    key_point = key_point.flatten()
    
    tree = cKDTree(original_mesh.points)
    
    _, key_point_idx = tree.query(key_point)
    key_point_on_mesh = original_mesh.points[key_point_idx]
    
    indices = tree.query_ball_point(key_point_on_mesh, r=radius)
    print(f"Found {len(indices)} vertices in neighborhood")
    return list(indices)

def process_side(pointcloud, key_point, region_points, original_mesh, path_width=None, extrude_distance=2.0, circle_indices=None):
    """
    Process one side (left or right): shortest path, region expansion, extrusion.

    Returns
    -------
    pv.PolyData
        Extruded mesh for the side.
    """
    graph = build_surface_graph_from_original_mesh(original_mesh, pointcloud)
    
    try:
        path_indices = find_shortest_path(graph, pointcloud.points, key_point, region_points)
    except ValueError as e:
        print(f"Error: {str(e)}")
        path_indices = []

    path_region_indices = create_path_region(pointcloud.points, path_indices, region_points, path_width=path_width)
    path_region_points = pointcloud.points[path_region_indices]
    
    circle_points = original_mesh.points[circle_indices]
    region_points = np.unique(np.vstack((path_region_points, circle_points)), axis=0)
    print(f"Merged region point count: {len(path_region_points)}")

    extruded_mesh = extrude_region(extrude_distance, original_mesh, region_points)
    
    return extruded_mesh

def generate_meshes(input_file, original_mesh_file, output_dir=None, path_width=None, extrude_height=None,
                   key_point_radius=5.0):
    """
    Generate left/right meshes from a labeled point cloud.

    Parameters
    ----------
    input_file : str
        TXT point cloud with labels.
    original_mesh_file : str
        Original mesh file.
    output_dir : str | None
        Output directory. Defaults to the input file directory.
    path_width : float
        Region expansion radius.
    extrude_height : float
        Extrusion distance.
    key_point_radius : float
        Neighborhood radius around the key point on the original mesh.

    Returns
    -------
    tuple[str, str, dict]
        (left_mesh_file, right_mesh_file, metadata)
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    pointcloud, left_key_point, right_key_point, left_region_points, right_region_points, original_mesh = load_point_cloud(input_file, original_mesh_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    left_mesh_file = os.path.join(output_dir, f"{base_name}_left_mesh.stl")
    right_mesh_file = os.path.join(output_dir, f"{base_name}_right_mesh.stl")
    
    left_circle_indices = create_key_point_circle(original_mesh, left_key_point, key_point_radius)
    right_circle_indices = create_key_point_circle(original_mesh, right_key_point, key_point_radius)
    
    left_mesh = process_side(pointcloud, left_key_point, left_region_points, original_mesh, path_width, extrude_height, left_circle_indices)
    left_mesh.save(left_mesh_file)
    print(f"Saved left mesh: {left_mesh_file}")
    
    right_mesh= process_side(pointcloud, right_key_point, right_region_points, original_mesh, path_width, extrude_height, right_circle_indices)
    right_mesh.save(right_mesh_file)
    print(f"Saved right mesh: {right_mesh_file}")
    
    return left_mesh_file, right_mesh_file, {"left_key_point": left_key_point, "right_key_point": right_key_point}


if __name__ == "__main__":
    input_file = os.path.join("data", "example", "prediction.txt")
    original_mesh_file = os.path.join("data", "example", "original_mesh.ply")
    output_dir = os.path.join("outputs", "stage1_meshes")

    path_width = 3.5
    extrude_height = 2.0
    key_point_radius = 6.0

    for p in [input_file, original_mesh_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required file: {p}. Place your data under data/example/ or update the paths in __main__."
            )

    left_mesh_file, right_mesh_file, _ = generate_meshes(
        input_file=input_file,
        original_mesh_file=original_mesh_file,
        output_dir=output_dir,
        path_width=path_width,
        extrude_height=extrude_height,
        key_point_radius=key_point_radius,
    )

    print(f"Done. Left: {left_mesh_file} | Right: {right_mesh_file}")
