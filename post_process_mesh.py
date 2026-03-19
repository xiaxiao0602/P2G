import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pyvista as pv
import trimesh
import pymeshfix as mf

from generate_meshes import load_point_cloud

class MeshPostProcessor:
    """
    Post-process meshes produced by generate_meshes.py.

    This module adds hollow cylinders and a bridging structure and attempts to ensure
    the final output is watertight.
    """
    
    def __init__(self, cylinder_radius: float = 1.0, 
                 cylinder_height: float = 10.0):
        """
        Parameters
        ----------
        cylinder_radius : float
            Cylinder radius.
        cylinder_height : float
            Cylinder height.
        """
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        
    def calculate_optimal_directions(self, json_file: str) -> Dict[str, Any]:
        """
        Load precomputed trajectory/direction information from a JSON file.

        The expected schema is documented in README.
        """
        print(f"Loading precomputed trajectory JSON: {json_file}")
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Trajectory JSON not found: {json_file}")
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            print(f"Loaded trajectory JSON: {json_file}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON: {e}")
    
    def create_hollow_cylinder(self, center: np.ndarray, direction: np.ndarray, 
                              outer_radius: float, inner_radius: float, 
                              height: float, label: int = None, side: str = None, 
                              output_dir: str = None) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        Create an outer and inner cylinder for subsequent boolean operations.
        """
        print(f"Creating hollow cylinder: center={center}, direction={direction}")
        
        direction = direction / np.linalg.norm(direction)
        
        outer_cylinder = trimesh.creation.cylinder(
            radius=outer_radius,
            height=height,
            sections=32,
            transform=trimesh.geometry.align_vectors([0, 0, 1], direction, return_angle=False)
        )
        outer_cylinder.apply_translation(center)
        
        inner_height = height * 1.1
        
        inner_cylinder = trimesh.creation.cylinder(
            radius=inner_radius,
            height=inner_height,
            sections=32,
            transform=trimesh.geometry.align_vectors([0, 0, 1], direction, return_angle=False)
        )
        inner_cylinder.apply_translation(center)

        return outer_cylinder, inner_cylinder

    def find_region_center(self, region_points, method='ellipse_fitting'):
        """
        Estimate a representative center point for a (roughly elliptical) region.
        """
        if method == 'centroid':
            return self._find_center_by_centroid(region_points)
        elif method == 'ellipse_fitting':
            return self._find_center_by_ellipse_fitting(region_points)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _find_center_by_centroid(self, region_points):
        """
        Simple centroid-based method.
        """
        centroid = np.mean(region_points, axis=0)
        
        distances = np.linalg.norm(region_points - centroid, axis=1)
        
        closest_idx = np.argmin(distances)
        closest_point = region_points[closest_idx]
        
        return closest_point
    
    def _find_center_by_ellipse_fitting(self, region_points):
        """
        Estimate center by fitting an ellipse in the region's PCA plane.
        """
        try:
            from sklearn.decomposition import PCA
            from scipy.optimize import minimize
            
            pca = PCA(n_components=2)
            
            points_2d = pca.fit_transform(region_points)
            
            def ellipse_residual(params, points):
                """Residual function for ellipse fitting."""
                cx, cy, a, b, theta = params
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                x_rot = (points[:, 0] - cx) * cos_theta + (points[:, 1] - cy) * sin_theta
                y_rot = -(points[:, 0] - cx) * sin_theta + (points[:, 1] - cy) * cos_theta
                
                residual = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
                return np.sum(residual ** 2)
            
            cx_init = np.mean(points_2d[:, 0])
            cy_init = np.mean(points_2d[:, 1])
            a_init = np.std(points_2d[:, 0]) * 2
            b_init = np.std(points_2d[:, 1]) * 2
            theta_init = 0
            
            initial_params = [cx_init, cy_init, a_init, b_init, theta_init]
            
            result = minimize(ellipse_residual, initial_params, args=(points_2d,), 
                            method='L-BFGS-B',
                            bounds=[(-np.inf, np.inf), (-np.inf, np.inf), 
                                   (0.1, np.inf), (0.1, np.inf), 
                                   (-np.pi, np.pi)])
            
            if result.success:
                center_2d = np.array([result.x[0], result.x[1]])
                
                center_3d = pca.inverse_transform(center_2d.reshape(1, -1))[0]
                
                distances = np.linalg.norm(region_points - center_3d, axis=1)
                closest_idx = np.argmin(distances)
                return region_points[closest_idx]
            else:
                print("Ellipse fitting failed, falling back to PCA method")
                return self._find_center_by_pca(region_points)
                
        except ImportError:
            print("scikit-learn not available, falling back to PCA method")
            return self._find_center_by_pca(region_points)
        except Exception as e:
            print(f"Ellipse fitting error: {e}. Falling back to PCA method")
            return self._find_center_by_pca(region_points)

    def create_cuboid(self, origin, width, length, height, orientation_matrix, y_extension_factor=0.1):
        """
        Create a cuboid oriented by an orthonormal basis.

        Returns
        -------
        pv.PolyData
            Cuboid mesh.
        np.ndarray
            One reference vertex (used downstream).
        """
        width = float(width)
        length = float(length)
        height = float(height)
        
        extension_height = height * y_extension_factor
        
        vertices_local = np.array([
            [0, extension_height, 0],
            [width, extension_height, 0],
            [width, extension_height, length],
            [0, extension_height, length],
            [0, -height * 1.001, 0],
            [width, -height * 1.001, 0],
            [width, -height * 1.001, length],
            [0, -height * 1.001, length]
        ])
        
        vertices_global = []
        for vertex in vertices_local:
            transformed = np.dot(orientation_matrix, vertex)
            vertices_global.append(origin + transformed)
        
        vertices_global = np.array(vertices_global)
        
        faces = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5]
        ]
        
        vtk_faces = []
        for face in faces:
            vtk_faces.extend([3, face[0], face[1], face[2]])
        
        cuboid = pv.PolyData(vertices_global, vtk_faces)
        
        return cuboid, vertices_global[7]
    
    def create_bridge_cuboid(self, left_mesh: trimesh.Trimesh, right_mesh: trimesh.Trimesh,
                            left_region_points: np.ndarray, right_region_points: np.ndarray,
                            original_mesh: trimesh.Trimesh,
                            left_key_point: np.ndarray, right_key_point: np.ndarray,
                            cuboid_width: float = 3.0, cuboid_length: float = 5.0) -> trimesh.Trimesh:
        """
        Create a bridging cuboid structure connecting left/right meshes.
        """
        print("创建桥接长方体，连接左右两侧mesh底部...")
        print("创建相对坐标系和长方体支撑结构...")

        left_center = self.find_region_center(left_region_points, method='ellipse_fitting')
        right_center = self.find_region_center(right_region_points, method='ellipse_fitting')

        
        point_A = left_center + (left_center - right_center)/16
        point_B = right_center + (right_center - left_center)/16
        point_C = (point_A + point_B) / 2
        
        left_key = left_key_point.flatten()
        right_key = right_key_point.flatten()
        point_D = (left_key + right_key) / 2
        
        print(f"点A (左区域中心): {point_A}")
        print(f"点B (右区域中心): {point_B}")
        print(f"点C (AB中点): {point_C}")
        print(f"点D (关键点中点): {point_D}")
        
        x_axis = point_B - point_A
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        z_axis = point_D - point_C
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        orientation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        print(f"X轴: {x_axis}")
        print(f"Y轴: {y_axis}")
        print(f"Z轴: {z_axis}")
        
        mesh_points = original_mesh.vertices
        
        local_coords = np.zeros_like(mesh_points)
        for i, point in enumerate(mesh_points):
            relative_vec = point - point_C
            local_coords[i, 0] = np.dot(relative_vec, x_axis)
            local_coords[i, 1] = np.dot(relative_vec, y_axis)
            local_coords[i, 2] = np.dot(relative_vec, z_axis)
        
        min_y_idx = np.argmin(local_coords[:, 1])
        min_y_value = local_coords[min_y_idx, 1]
        min_y_point = mesh_points[min_y_idx]
        
        print(f"Y坐标最小的点: {min_y_point}, Y值: {min_y_value}")
        
        cuboid_height = abs(min_y_value)
        print(f"长方体高度: {cuboid_height}") 
        
        left_cuboid, left_bridge_back = self.create_cuboid(
            origin=point_A,  
            width=cuboid_width,
            length=cuboid_length,
            height=cuboid_height,
            orientation_matrix=orientation_matrix,
            y_extension_factor=0.2
        )
        
        right_cuboid, right_bridge_back = self.create_cuboid(
            origin=point_B,  
            width=cuboid_width,
            length=cuboid_length,
            height=cuboid_height,
            orientation_matrix=orientation_matrix,
            y_extension_factor=0.2
        )


        
        print("创建桥接长方体，连接左右两侧长方体底部...")
        
        left_cuboid_points = left_cuboid.points
        right_cuboid_points = right_cuboid.points
        
        left_points_local = []
        for point in left_cuboid_points:
            relative_vec = point - point_C
            x_local = np.dot(relative_vec, x_axis)
            y_local = np.dot(relative_vec, y_axis)
            z_local = np.dot(relative_vec, z_axis)
            left_points_local.append([x_local, y_local, z_local, point[0], point[1], point[2]])
        
        right_points_local = []
        for point in right_cuboid_points:
            relative_vec = point - point_C
            x_local = np.dot(relative_vec, x_axis)
            y_local = np.dot(relative_vec, y_axis)
            z_local = np.dot(relative_vec, z_axis)
            right_points_local.append([x_local, y_local, z_local, point[0], point[1], point[2]])
        
        left_points_local = np.array(left_points_local)
        right_points_local = np.array(right_points_local)
        

        min_y_left = np.min(left_points_local[:, 1])
        min_y_right = np.min(right_points_local[:, 1])
        
        bottom_left = left_points_local[np.isclose(left_points_local[:, 1], min_y_left, atol=1e-5)]
        bottom_right = right_points_local[np.isclose(right_points_local[:, 1], min_y_right, atol=1e-5)]
        
        if len(bottom_left) < 4 or len(bottom_right) < 4:
            print(f"警告: 底面顶点识别不完整，左:{len(bottom_left)}个, 右:{len(bottom_right)}个")
            if len(bottom_left) < 2 or len(bottom_right) < 2:
                print("无法创建桥接长方体，底面顶点太少")
            else:
                print("尝试使用可用的顶点创建桥接长方体...")
        
        if len(bottom_left) >= 2:
            sorted_left = bottom_left[np.argsort(bottom_left[:, 0])]
            left_min_x_points = sorted_left[:2, 3:6]
        else:
            print("左侧底面点不足，无法创建桥接长方体")
            left_min_x_points = []
        
        if len(bottom_right) >= 2:
            sorted_right = bottom_right[np.argsort(bottom_right[:, 0])]
            right_max_x_points = sorted_right[-2:, 3:6]
        else:
            print("右侧底面点不足，无法创建桥接长方体")
            right_max_x_points = []
        
        if len(left_min_x_points) == 2 and len(right_max_x_points) == 2:
            left_min_x_points = left_min_x_points[np.argsort(left_min_x_points[:, 2])]
            right_max_x_points = right_max_x_points[np.argsort(right_max_x_points[:, 2])]
            
            bridge_height = cuboid_width
                        
            y_offset = -y_axis * bridge_height

            bridge_points = np.vstack([
                left_min_x_points[0] - y_offset * 0.001,
                right_max_x_points[0] - y_offset * 0.001,
                right_max_x_points[1] - y_offset * 0.001,
                left_min_x_points[1] - y_offset * 0.001,
            ])
            
            bottom_points = np.vstack([
                bridge_points[0] + y_offset,
                bridge_points[1] + y_offset,
                bridge_points[2] + y_offset,
                bridge_points[3] + y_offset,
            ])
            
            all_bridge_points = np.vstack([bridge_points, bottom_points])
            
            bridge_faces = [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [3, 2, 6],
                [3, 6, 7],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2]
            ]
            
            vtk_faces = []
            for face in bridge_faces:
                vtk_faces.extend([3, face[0], face[1], face[2]])
            
            bridge_cuboid = pv.PolyData(all_bridge_points, vtk_faces)
            
            bridge_cuboid = bridge_cuboid.clean(tolerance=1e-10)
            
            return left_cuboid, right_cuboid, bridge_cuboid


    
    def ensure_watertight(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Attempt to make a mesh watertight and a valid volume.

        Steps:
        1) Try Trimesh built-in repairs.
        2) If needed, fall back to pymeshfix.
        """
        print("开始确保网格成为有效体积...")
        
        if mesh.is_watertight and mesh.is_volume:
            print("网格已经是有效体积。")
            return mesh

        print("第一轮：尝试使用trimesh内置修复...")
        try:
            mesh.process()
            mesh.merge_vertices()
            
            mesh.update_faces(mesh.nondegenerate_faces())
            
            mesh.update_faces(mesh.unique_faces())
            mesh.fill_holes()
            mesh.fix_normals()
        except Exception as e:
            print(f"Trimesh内置修复时出错: {e}")

        if mesh.is_volume:
            print("Trimesh内置修复成功，网格是有效体积。")
            return mesh
        
        print("Trimesh内置修复失败，升级到pymeshfix...")

        repaired_mesh = None
        try:
            v, f = mesh.vertices, mesh.faces
            mfix = mf.MeshFix(v, f)
            mfix.repair()
            repaired_mesh = trimesh.Trimesh(mfix.v, mfix.f, process=False)
            repaired_mesh.fix_normals()
            
            if repaired_mesh and repaired_mesh.is_volume:
                print("Pymeshfix修复成功，网格是有效体积。")
                return repaired_mesh
            else:
                print("Pymeshfix修复后仍不是有效体积。")
        except Exception as e:
            print(f"Pymeshfix修复过程中出现错误: {e}")

        
    
    def fix_mesh_normals(self, mesh):
        """
        Compute normals for a PyVista mesh and guard against invalid results.
        """
        try:
            if mesh.n_faces == 0:
                print("警告: mesh没有面片，无法计算法向量")
                return mesh
            
            mesh.compute_normals(inplace=True, point_normals=True, cell_normals=True)
            
            if 'Normals' not in mesh.point_data:
                print("警告: 法向量计算失败，使用默认法向量")
                default_normals = np.zeros((mesh.n_points, 3))
                default_normals[:, 2] = 1.0
                mesh.point_data['Normals'] = default_normals
            else:
                normals = mesh.point_data['Normals']
                if np.any(np.isnan(normals)) or normals is None:
                    print("警告: 法向量包含无效值，使用默认法向量")
                    default_normals = np.zeros((mesh.n_points, 3))
                    default_normals[:, 2] = 1.0
                    mesh.point_data['Normals'] = default_normals
            
            return mesh
            
        except Exception as e:
            print(f"修复法向量时出错: {e}")
            return mesh
    
    def load_mesh_from_file(self, mesh_file: str) -> trimesh.Trimesh:
        """
        Load a mesh from file as a trimesh.Trimesh.
        """
        print(f"加载mesh文件: {mesh_file}")
        
        try:
            mesh = trimesh.load(mesh_file)
            
            if isinstance(mesh, trimesh.Scene):
                mesh = list(mesh.geometry.values())[0]
            
            return mesh
        except Exception as e:
            print(f"使用trimesh加载失败: {e}，尝试使用pyvista")
            
            pv_mesh = pv.read(mesh_file)
            
            vertices = pv_mesh.points
            faces = []
            
            for i in range(pv_mesh.n_cells):
                cell = pv_mesh.get_cell(i)
                if cell.type == 5:  # VTK_TRIANGLE
                    faces.append(cell.point_ids)
            
            if len(faces) == 0:
                raise ValueError("未找到有效的三角形面")
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
    
    def process_mesh(self, left_mesh_file: str, right_mesh_file: str, original_mesh_file: str,
                    left_region_points: np.ndarray, right_region_points: np.ndarray,
                    direction_result: Dict[str, Any], output_dir: str = None, 
                    cylinder_radius: float = 1.0, cylinder_height: float = 5.0, 
                    cuboid_height: float = 1.0) -> str:
        """
        Post-process left/right meshes into a single guide plate mesh.

        Parameters
        ----------
        direction_result : dict
            Trajectory/direction info loaded from JSON.
        """
        print("开始处理mesh...")
        
        if output_dir is None:
            output_dir = os.path.dirname(left_mesh_file)
        
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.cuboid_height = cuboid_height
        
        left_mesh = self.load_mesh_from_file(left_mesh_file)
        right_mesh = self.load_mesh_from_file(right_mesh_file)
        original_mesh = self.load_mesh_from_file(original_mesh_file)
        
        label = None
        try:
            base_name = os.path.basename(left_mesh_file)
            if '_mask_' in base_name:
                label = int(base_name.split('_mask_')[0])
        except (ValueError, IndexError):
            print("无法从文件名中提取label信息")
        
        processed_left, left_key_point = self._process_single_mesh(left_mesh, direction_result, 'Left', label, output_dir)
        
        processed_right, right_key_point = self._process_single_mesh(right_mesh, direction_result, 'Right', label, output_dir)
        
        left_cuboid, right_cuboid, bridge_cuboid = self.create_bridge_cuboid(
            left_mesh=left_mesh,
            right_mesh=right_mesh,
            left_region_points=left_region_points,
            right_region_points=right_region_points,
            original_mesh=original_mesh,
            left_key_point=left_key_point,
            right_key_point=right_key_point
        )
        
        print("合并左右mesh和桥接长方体...")

        meshes_to_check = [
            ("左侧长方体", left_cuboid),
            ("右侧长方体", right_cuboid),
            ("左侧mesh", processed_left),
            ("右侧mesh", processed_right),
            ("桥接长方体", bridge_cuboid)
        ]
        
        for mesh_name, mesh_obj in meshes_to_check:
            if mesh_obj is not None:
                if not isinstance(mesh_obj, trimesh.Trimesh):
                    print(f"{mesh_name}不是trimesh对象，转换为trimesh对象...")
                    if hasattr(mesh_obj, 'points') and hasattr(mesh_obj, 'faces'):
                        mesh_obj = trimesh.Trimesh(vertices=mesh_obj.points, faces=mesh_obj.faces.reshape(-1, 4)[:, 1:4])
                    else:
                        print(f"警告: {mesh_name}无法转换为trimesh对象，跳过检查...")
                        continue
                
                if not mesh_obj.is_volume:
                    print(f"{mesh_name}不是封闭体积，尝试修复...")
                    mesh_obj = self.ensure_watertight(mesh_obj)
                    print(f"修复后{mesh_name} is_volume: {mesh_obj.is_volume}, is_watertight: {mesh_obj.is_watertight}")
                
                if mesh_name == "左侧长方体":
                    left_cuboid = mesh_obj
                elif mesh_name == "右侧长方体":
                    right_cuboid = mesh_obj
                elif mesh_name == "左侧mesh":
                    processed_left = mesh_obj
                elif mesh_name == "右侧mesh":
                    processed_right = mesh_obj
                elif mesh_name == "桥接长方体":
                    bridge_cuboid = mesh_obj
        

        print("开始合并所有组件...")
        meshes_to_combine = [
            processed_left,
            left_cuboid,
            bridge_cuboid,
            right_cuboid,
            processed_right
        ]
        
        meshes_to_combine = [m for m in meshes_to_combine if m is not None and isinstance(m, trimesh.Trimesh)]
        
        if not all(m.is_volume for m in meshes_to_combine):
            print("警告: 不是所有用于合并的mesh都是有效体积，正在逐一检查...")
            for i, m in enumerate(meshes_to_combine):
                if not m.is_volume:
                    print(f"  - Mesh {i} is not a volume.")

        try:
            combined_mesh = trimesh.boolean.union(meshes_to_combine, engine='manifold')
            print("所有组件合并完成。")
        except Exception as e:
            print(f"使用manifold引擎合并失败: {e}")

        combined_mesh = self.ensure_watertight(combined_mesh)

        original_mesh = self.ensure_watertight(original_mesh)
        combined_mesh = combined_mesh.difference(original_mesh)
        combined_mesh = self.ensure_watertight(combined_mesh)
        
        base_name = os.path.splitext(os.path.basename(left_mesh_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_guide_plate.stl")
        
        combined_mesh.export(output_file)
        
        print(f"完整导板处理完成: {output_file}")
        
        return output_file
    
    def _process_single_mesh(self, mesh: trimesh.Trimesh, 
                           direction_result: Dict[str, Any], 
                           side: str, label: int = None, output_dir: str = None) -> trimesh.Trimesh:
        """
        Add hollow cylinders to a single side mesh based on direction_result.
        """
        print(f"处理{side}侧mesh")
        
        target_points = []
        for point_info in direction_result.get('formatted_origin', []):
            if point_info.get('Line') == side:
                target_point = np.array([
                    point_info['Entry_X'],
                    point_info['Entry_Y'],
                    point_info['Entry_Z']
                ])
                direction = np.array([
                    point_info['Direction_X'],
                    point_info['Direction_Y'],
                    point_info['Direction_Z']
                ])
                target_points.append((target_point, direction))
        
        if not target_points:
            print(f"警告: 未找到{side}侧的目标点")
            return self.ensure_watertight(mesh)
        

        combined_mesh = mesh.copy()
        
        if not combined_mesh.is_volume:
            print(f"警告: 初始{side}侧mesh不是有效的体积mesh")
            print(f"初始mesh信息: 顶点数={len(combined_mesh.vertices)}, 面数={len(combined_mesh.faces)}, 是否水密={combined_mesh.is_watertight}")
            combined_mesh = self.ensure_watertight(combined_mesh)
            if combined_mesh.is_volume:
                print("初始mesh修复成功，现在是有效体积")
            else:
                print("初始mesh修复失败，仍不是有效体积，这可能导致后续布尔运算失败")
        
        for target_point, direction in target_points:
            
            outer_cylinder, inner_cylinder = self.create_hollow_cylinder(
                center=target_point,
                direction=direction,
                outer_radius=self.cylinder_radius + 1.5,
                inner_radius=self.cylinder_radius,
                height=self.cylinder_height,
                label=label,
                side=side,
                output_dir=output_dir
            )
            
            outer_cylinder = self.ensure_watertight(outer_cylinder)
            combined_mesh = combined_mesh.union(outer_cylinder)
            combined_mesh = self.ensure_watertight(combined_mesh)

            combined_mesh = combined_mesh.difference(inner_cylinder)

            print("成功添加空心圆柱")
        
        return combined_mesh, target_point

def post_process_meshes(left_mesh_file: str, right_mesh_file: str, original_mesh_file: str,
                        left_region_points: np.ndarray, right_region_points: np.ndarray,
                        json_file: str, output_dir: str = None, 
                        cylinder_radius: float = 1.5, cylinder_height: float = 5.0,
                        cuboid_height: float = 1.0) -> str:
    """
    Convenience wrapper around MeshPostProcessor.process_mesh.
    """
    print("开始处理已有的左右mesh文件...")
    
    if output_dir is None:
        output_dir = os.path.dirname(left_mesh_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON文件不存在: {json_file}")
    
    print(f"读取JSON文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        direction_result = json.load(f)
    
    processor = MeshPostProcessor(
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height
    )
    
    try:
        output_file = processor.process_mesh(
            left_mesh_file=left_mesh_file,
            right_mesh_file=right_mesh_file,
            left_region_points=left_region_points,
            right_region_points=right_region_points,
            original_mesh_file=original_mesh_file,
            direction_result=direction_result,
            output_dir=output_dir,
            cylinder_radius=cylinder_radius,
            cylinder_height=cylinder_height,
            cuboid_height=cuboid_height,
        )
        
        return output_file
        
    except Exception as e:
        print(f"处理mesh时出错: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    input_file = os.path.join("data", "example", "prediction.txt")
    left_mesh_file = os.path.join("outputs", "stage1_meshes", "prediction_left_mesh.stl")
    right_mesh_file = os.path.join("outputs", "stage1_meshes", "prediction_right_mesh.stl")
    original_mesh_file = os.path.join("data", "example", "original_mesh.ply")
    json_file = os.path.join("data", "example", "trajectory.json")
    output_dir = os.path.join("outputs", "stage2_guide_plate")

    for p in [input_file, left_mesh_file, right_mesh_file, original_mesh_file, json_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required file: {p}. Place your data under data/example/ and outputs/ or update the paths in __main__."
            )

    os.makedirs(output_dir, exist_ok=True)

    _, _, _, left_region_points, right_region_points, _ = load_point_cloud(input_file)
    output_file = post_process_meshes(
        left_mesh_file=left_mesh_file,
        right_mesh_file=right_mesh_file,
        left_region_points=left_region_points,
        right_region_points=right_region_points,
        original_mesh_file=original_mesh_file,
        json_file=json_file,
        output_dir=output_dir,
        cylinder_radius=0.75,
        cylinder_height=20.0,
        cuboid_height=1.0,
    )

    print(f"Guide plate saved: {output_file}")
