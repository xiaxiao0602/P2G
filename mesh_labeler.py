import os
import json
import numpy as np
import pyvista as pv
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from collections import deque
import tkinter as tk
from tkinter import filedialog
import sys
import time

class BasicMeshLabeler:
    def __init__(self):
        print("Initializing PLY File Labeling Tool...")
        
        # Initialize PyVista
        self.plotter = pv.Plotter()
        
        # State variables
        self.mesh = None
        self.mesh_file_path = None  # Store the path of loaded mesh file
        self.point_labels = {}  # Point labels {point index: label name}
        self.region_labels = []  # Region labels [{region name: str, point indices: List[int]}]
        self.selecting_region = False
        self.temp_region_points = []
        self.current_label = "Key Point"
        self.current_region_label = "Surface Region"
        self.pick_tolerance = 0.005  # Point picking tolerance
        self.point_size = 0.2  # Increased size of the labeled points
        self.last_pick_time = 0  # Time of last pick to prevent double picks
        self.pick_debounce_time = 0.5  # Time in seconds to wait between picks
        self.is_picking_enabled = False  # Flag to track if picking is enabled
        self.mesh_opacity = 0.6
        
        self.history = []
        self.current_history_index = -1
        
        self.ply_files = []
        self.current_file_index = -1
        
        # Set title and background
        self.plotter.set_background("gray")
        
        # Set keyboard events
        self.setup_key_events()
        
        # Add instruction text
        self.add_instructions()
        
        # Show file selection dialog
        self.show_file_dialog()
        
    def setup_key_events(self):
        """Set up keyboard events"""
        self.plotter.add_key_event('p', self.toggle_point_mode)
        self.plotter.add_key_event('r', self.toggle_region_mode)
        self.plotter.add_key_event('f', self.finish_region)
        self.plotter.add_key_event('c', self.clear_selection)
        self.plotter.add_key_event('s', self.save_labels)
        self.plotter.add_key_event('o', self.show_file_dialog)
        self.plotter.add_key_event('space', self.undo)
        self.plotter.add_key_event('t', self.toggle_transparency)
        self.plotter.add_key_event('q', self.load_next_file)
        self.plotter.add_key_event('m', self.exit_program)
        
    def add_instructions(self):
        """Add operation instructions"""
        instructions = [
            "Keyboard Controls:",
            "P: Point Labeling Mode",
            "R: Region Labeling Mode",
            "F: Finish Region",
            "C: Clear Selection",
            "S: Save Labels",
            "O: Open New Mesh File",
            "Q: Load Next PLY File",
            "Space: Undo Last Action",
            "T: Toggle Transparency",
            "M: Exit Program"
        ]
        
        self.plotter.add_text("\n".join(instructions), position="upper_left", font_size=8)
        
        # Add mode text
        mode_text = "Current Mode: "
        mode_text += "Point Labeling" if not self.selecting_region else "Region Labeling"
        self.plotter.add_text(mode_text, position="lower_left", font_size=8)
        
    def reset_picking(self):
        """Reset picking to avoid accidental picks"""
        if self.is_picking_enabled:
            self.plotter.disable_picking()
            self.is_picking_enabled = False
            time.sleep(0.1)  # Small delay to ensure picking is disabled
    
    def enable_picking(self):
        """Enable point picking with proper settings"""
        # Make sure picking is disabled first
        self.reset_picking()
        
        # Enable picking with our callback
        self.plotter.enable_point_picking(
            callback=self.on_point_picked, 
            show_message=False,
            use_picker=True,
            show_point=False,  # Don't show PyVista's own point, we'll render our own
            tolerance=self.pick_tolerance
        )
        self.is_picking_enabled = True
        
    def toggle_point_mode(self):
        """Switch to point labeling mode"""
        if self.selecting_region:
            camera_state = {
                'position': self.plotter.camera_position,
                'focal_point': self.plotter.camera.focal_point,
                'view_up': self.plotter.camera.up,
                'view_angle': self.plotter.camera.view_angle,
                'parallel_scale': self.plotter.camera.parallel_scale
            }
            
            self.selecting_region = False
            self.reset_picking()
            time.sleep(0.1)
            self.enable_picking()
            print("Switched to point labeling mode")
            
            self.update_visualization()
            self.plotter.camera_position = camera_state['position']
            self.plotter.camera.focal_point = camera_state['focal_point']
            self.plotter.camera.up = camera_state['view_up']
            self.plotter.camera.view_angle = camera_state['view_angle']
            self.plotter.camera.parallel_scale = camera_state['parallel_scale']
            self.plotter.render()
        
    def toggle_region_mode(self):
        """Switch to region labeling mode"""
        # Only change mode if not already in region mode
        if not self.selecting_region:
            camera_state = {
                'position': self.plotter.camera_position,
                'focal_point': self.plotter.camera.focal_point,
                'view_up': self.plotter.camera.up,
                'view_angle': self.plotter.camera.view_angle,
                'parallel_scale': self.plotter.camera.parallel_scale
            }
            
            self.selecting_region = True
            self.temp_region_points = []
            self.reset_picking()
            time.sleep(0.3)
            self.enable_picking()
            print("Switched to region labeling mode")
            
            self.update_visualization()
            self.plotter.camera_position = camera_state['position']
            self.plotter.camera.focal_point = camera_state['focal_point']
            self.plotter.camera.up = camera_state['view_up']
            self.plotter.camera.view_angle = camera_state['view_angle']
            self.plotter.camera.parallel_scale = camera_state['parallel_scale']
            self.plotter.render()
    
    def on_point_picked(self, point, _):
        """Point picking callback function
        Args:
            point: Picked point coordinates
        """
        # Debounce picking to prevent double picks
        current_time = time.time()
        if current_time - self.last_pick_time < self.pick_debounce_time:
            return
        self.last_pick_time = current_time
        
        if self.mesh is None:
            return
            
        # Find nearest point index
        point_idx = self.find_nearest_point(point)
        
        if point_idx is not None:
            if self.selecting_region:
                # Region labeling mode
                if point_idx not in self.temp_region_points:
                    self.temp_region_points.append(int(point_idx))
                    print(f"Selected point {point_idx}")
                    self.add_to_history('region_point', {'point_idx': int(point_idx)})
                    self.update_visualization(incremental=True, new_point_idx=int(point_idx))
            else:
                # Point labeling mode
                self.point_labels[int(point_idx)] = self.current_label
                print(f"Selected point {point_idx}")
                self.add_to_history('point_label', {'point_idx': int(point_idx), 'label': self.current_label})
                self.update_visualization(incremental=False)
        else:
            print("No valid point found")
    
    def find_nearest_point(self, point):
        """Find nearest point index"""
        if self.mesh is None:
            return None
            
        # Get mesh points
        mesh_points = self.mesh.points
        
        # Calculate distance from all points to target point
        distances = np.linalg.norm(mesh_points - point, axis=1)
        
        # Get camera position and direction
        camera_pos = np.array(self.plotter.camera_position[0])
        camera_dir = np.array(self.plotter.camera_position[1]) - camera_pos
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        # Find points within tolerance
        valid_indices = np.where(distances <= self.pick_tolerance)[0]
        
        if len(valid_indices) == 0:
            return None
            
        # Check visibility and occlusion for each valid point
        visible_points = []
        for idx in valid_indices:
            point_pos = mesh_points[idx]
            # Calculate vector from camera to point
            point_dir = point_pos - camera_pos
            point_dir = point_dir / np.linalg.norm(point_dir)
            
            # Calculate dot product to check if point is in front of camera
            dot_product = np.dot(camera_dir, point_dir)
            
            # If dot product is positive, point is in front of camera
            if dot_product > 0:
                # Check for occlusion using ray tracing
                # Create a ray from camera to point
                ray_start = camera_pos
                ray_end = point_pos
                
                # Perform ray tracing
                points, ind = self.mesh.ray_trace(ray_start, ray_end)
                
                # If the ray hits the point directly (no other intersections), the point is visible
                if len(points) == 1 and np.allclose(points[0], point_pos, atol=1e-6):
                    visible_points.append(idx)
        
        if len(visible_points) == 0:
            return None
            
        # Among visible points, find the closest one
        visible_distances = distances[visible_points]
        closest_idx = visible_points[np.argmin(visible_distances)]
        
        return int(closest_idx)  # Convert to regular Python integer
    
    def get_points_in_region(self, boundary_points):
        """Get all points in region using shortest paths between boundary points"""
        if len(boundary_points) < 3:
            return []
        
        print("Calculating region points, please wait...")
        
        try:
            surf = self.mesh.extract_surface()
            
            faces = surf.faces.reshape(-1, 4)[:, 1:4]
            
            edges = []
            for face in faces:
                edges.extend([(face[0], face[1]), (face[1], face[2]), (face[2], face[0])])
            
            edges = list(set(tuple(sorted(edge)) for edge in edges))
            
            weights = []
            for edge in edges:
                p1 = surf.points[edge[0]]
                p2 = surf.points[edge[1]]
                weight = np.linalg.norm(p1 - p2)
                weights.append(weight)
            
            n_points = len(surf.points)
            row = [e[0] for e in edges] + [e[1] for e in edges]
            col = [e[1] for e in edges] + [e[0] for e in edges]
            data = weights + weights
            graph = csr_matrix((data, (row, col)), shape=(n_points, n_points))
            
            boundary_paths = []
            path_points = set()
            
            for i in range(len(boundary_points)):
                start_idx = boundary_points[i]
                end_idx = boundary_points[(i + 1) % len(boundary_points)]
                
                dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, 
                                                       indices=start_idx, 
                                                       return_predecessors=True)
                
                path = []
                current = end_idx
                while current != start_idx:
                    path.append(current)
                    current = predecessors[current]
                path.append(start_idx)
                path.reverse()
                
                boundary_paths.append(path)
                path_points.update(path)
            
            points_in_region = list(path_points)
            
            centroid = np.mean(surf.points[list(path_points)], axis=0)
            distances = np.linalg.norm(surf.points - centroid, axis=1)
            start_point = np.argmin(distances)
            
            queue = deque([start_point])
            visited = set(path_points)
            
            while queue:
                current = queue.popleft()
                points_in_region.append(current)
                
                neighbors = np.nonzero(graph[current].toarray()[0])[0]
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            print(f"Found {len(points_in_region)} points in region")
            return points_in_region
            
        except Exception as e:
            print(f"Failed to calculate points in region: {str(e)}")
            print("Please try selecting more boundary points to form a better closed region")
            return []
            
    def ray_intersects_segment(self, point, p1, p2):
        """Check if a horizontal ray from point intersects the line segment p1-p2"""
        # Check if point is between p1 and p2 in y-coordinate
        if (p1[1] > point[1]) == (p2[1] > point[1]):
            return False
            
        # Check if point is to the right of the line segment
        x = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
        return point[0] < x
    
    def finish_region(self):
        """Complete region labeling"""
        if len(self.temp_region_points) < 3:
            print("Region requires at least 3 points")
            return
        
        # Disable picking during region calculation
        self.reset_picking()
            
        # Get all points in region
        region_points = self.get_points_in_region(self.temp_region_points)
        
        # Ensure all point indices are regular Python integers
        points_list = [int(idx) for idx in region_points]
        
        self.add_to_history('finish_region', {
            'name': self.current_region_label,
            'points': points_list,
            'boundary_points': self.temp_region_points.copy()
        })
        
        self.region_labels.append({
            "name": self.current_region_label,
            "points": points_list,
            "boundary_points": self.temp_region_points.copy()
        })
        
        print(f"Region completed with {len(points_list)} points")
        self.temp_region_points = []
        
        self.update_visualization(incremental=False)
        self.selecting_region = False
        
        self.save_labels()
    
    def clear_selection(self):
        """Clear current selection"""
        self.temp_region_points = []
        self.update_visualization()
        print("Selection cleared")
    
    def update_visualization(self, incremental=False, new_point_idx=None):
        """Update visualization
        Args:
            incremental: 是否为增量更新
            new_point_idx: 新添加的点索引
        """
        if self.mesh is None:
            return
            
        if not incremental:
            # Re-display mesh
            self.plotter.clear()
            
            self.mesh.compute_normals(inplace=True)
            
            normals = self.mesh.point_normals
            scalars = normals[:, 1]
            
            self.plotter.add_mesh(
                self.mesh,
                color='#f7f5d9',
                show_edges=True,
                opacity=self.mesh_opacity,
                smooth_shading=True,
                show_scalar_bar=False
            )
            
            # Re-add instructions
            self.add_instructions()
            
            # Display point labels - red
            for point_idx, label in self.point_labels.items():
                point = self.mesh.points[point_idx]
                # Create a larger sphere to represent labeled point
                sphere = pv.Sphere(center=point, radius=self.point_size * 2)
                self.plotter.add_mesh(sphere, color="red", render=True, opacity=1.0, smooth_shading=True)
                # Add label with offset
                label_point = point + np.array([self.point_size*2, self.point_size*2, self.point_size*2])
                self.plotter.add_point_labels([label_point], [label], font_size=12, bold=True)
            
            # Display region labels - with different colors for boundary points
            colors = [(0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
            for i, region in enumerate(self.region_labels):
                boundary_color = colors[i % len(colors)]
                region_color = (0.3, 0.7, 1.0)
                
                if len(region["points"]) > 0:
                    points = self.mesh.points[region["points"]]
                    point_cloud = pv.PolyData(points)
                    sphere = pv.Sphere(radius=self.point_size * 0.8, phi_resolution=8, theta_resolution=8)
                    glyphs = point_cloud.glyph(scale=1.0, geom=sphere, orient=False)
                    self.plotter.add_mesh(glyphs, color=region_color, opacity=0.8, smooth_shading=True)
                
                if len(region.get("boundary_points", [])) > 0:
                    boundary_points = self.mesh.points[region["boundary_points"]]
                    boundary_cloud = pv.PolyData(boundary_points)
                    sphere = pv.Sphere(radius=self.point_size * 1.5, phi_resolution=8, theta_resolution=8)
                    boundary_glyphs = boundary_cloud.glyph(scale=1.0, geom=sphere, orient=False)
                    self.plotter.add_mesh(boundary_glyphs, color=boundary_color, opacity=1.0, smooth_shading=True)
            
            # Display temporary region points - cyan
            if len(self.temp_region_points) > 0:
                temp_points = self.mesh.points[self.temp_region_points]
                temp_cloud = pv.PolyData(temp_points)
                sphere = pv.Sphere(radius=self.point_size * 1.5, phi_resolution=8, theta_resolution=8)
                temp_glyphs = temp_cloud.glyph(scale=1.0, geom=sphere, orient=False)
                self.plotter.add_mesh(temp_glyphs, color="cyan", opacity=1.0, smooth_shading=True)
        else:
            if new_point_idx is not None and new_point_idx in self.temp_region_points:
                point = self.mesh.points[new_point_idx]
                sphere = pv.Sphere(center=point, radius=self.point_size * 1.5)
                self.plotter.add_mesh(sphere, color="cyan", render=True, opacity=1.0, smooth_shading=True)
                
        # Force render update
        self.plotter.render()
    
    def show_file_dialog(self):
        """Show file selection dialog to load a PLY file"""
        # Hide the plotter window temporarily to show file dialog
        if hasattr(self.plotter, 'window') and self.plotter.window is not None:
            was_visible = self.plotter.window.IsVisible()
            if was_visible:
                self.plotter.window.SetWindowDisplayMode(0)  # Hide temporarily
        
        # Create root Tk window and hide it
        root = tk.Tk()
        root.withdraw()
        
        # Show file dialog
        file_path = filedialog.askopenfilename(
            title="Select PLY File",
            filetypes=[("PLY Files", "*.ply"), ("All Files", "*.*")]
        )
        
        # Destroy the root window
        root.destroy()
        
        # Restore plotter window
        if hasattr(self.plotter, 'window') and self.plotter.window is not None and was_visible:
            self.plotter.window.SetWindowDisplayMode(1)  # Show again
        
        # If a file was selected, load it
        if file_path:
            self.mesh_file_path = file_path
            self.load_mesh_file(file_path)
        else:
            # If no file was selected and we have no mesh, load example model
            if self.mesh is None:
                self.load_example_model()
                
    def load_mesh_file(self, file_path):
        """Load mesh file"""
        try:
            print(f"Loading mesh file: {file_path}")
            
            self.update_ply_files_list(file_path)
            
            mesh = pv.read(file_path)
            
            print(f"Mesh info: {mesh}")
            print(f"Number of points: {mesh.n_points}")
            print(f"Number of cells: {mesh.n_cells}")
            
            self.display_mesh(mesh, file_path)
            
            return True
        except Exception as e:
            print(f"Error loading mesh file: {str(e)}")
            return False
        
    def load_example_model(self):
        """Load example model"""
        # Load bunny model
        print("Loading example bunny model...")
        mesh = pv.examples.download_bunny()
        self.mesh_file_path = None  # No file path for example model
        self.display_mesh(mesh)
        
    def display_mesh(self, mesh, file_path=None):
        """Display mesh"""
        self.mesh = mesh
        if file_path:
            self.mesh_file_path = file_path
        
        # Clear and add mesh
        self.plotter.clear()
        
        mesh.compute_normals(inplace=True)
        
        normals = mesh.point_normals
        scalars = normals[:, 1]
        
        self.plotter.add_mesh(
            mesh,
            color='#f7f5d9',
            show_edges=True,
            opacity=self.mesh_opacity,
            smooth_shading=True,
            show_scalar_bar=False
        )
        
        self.plotter.reset_camera()
        
        # Re-add instructions
        self.add_instructions()
        
        # Reset picking state and enable picking
        self.reset_picking()
        self.enable_picking()
        
        # Update visualization to show any existing labels
        self.update_visualization()
        
    def save_labels(self):
        """Save labels"""
        if self.mesh is None:
            print("No mesh to save labels for")
            return
            
        try:
            # Convert point labels to serializable format
            point_labels_serializable = {}
            for point_idx, label in self.point_labels.items():
                point_labels_serializable[str(point_idx)] = label
                
            # Ensure point indices in region labels are serializable
            region_labels_serializable = []
            for region in self.region_labels:
                region_serializable = {
                    "name": region["name"],
                    "points": [int(idx) for idx in region["points"]],
                    "boundary_points": [int(idx) for idx in region["boundary_points"]]
                }
                region_labels_serializable.append(region_serializable)
                
            data = {
                "point_labels": point_labels_serializable,
                "region_labels": region_labels_serializable
            }
            
            # Determine where to save the file
            if self.mesh_file_path:
                # Save in the same directory as the PLY file with the same name but .json extension
                dir_name = os.path.dirname(self.mesh_file_path)
                base_name = os.path.splitext(os.path.basename(self.mesh_file_path))[0]
                save_path = os.path.join(dir_name, base_name + ".json")
            else:
                # Default path for example model
                save_path = "mesh_labels.json"
                
            # Save to the chosen path
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"Labels saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save labels: {str(e)}")
            
    def run(self):
        """Run program"""
        print("Starting labeling tool...")
        print("Press P for point labeling mode")
        print("Press R for region labeling mode")
        print("Press F to finish current region labeling")
        print("Press C to clear current selection")
        print("Press S to save labels")
        print("Press O to open a new mesh file")
        print("Press Q to load next PLY file")
        print("Press T to toggle transparency")
        print("Press Space to undo last action")
        try:
            self.plotter.show(auto_close=False)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            sys.exit(1)

    def add_to_history(self, action_type, data):
        """Append an action to the undo history."""
        if self.current_history_index < len(self.history) - 1:
            self.history = self.history[:self.current_history_index + 1]
        
        new_history = {
            'type': action_type,
            'data': data,
            'point_labels': self.point_labels.copy(),
            'region_labels': [r.copy() for r in self.region_labels],
            'temp_region_points': self.temp_region_points.copy()
        }
        
        self.history.append(new_history)
        self.current_history_index = len(self.history) - 1

    def undo(self):
        """Undo the last action."""
        if self.current_history_index < 0:
            print("No more actions to undo")
            return
            
        self.current_history_index -= 1
        
        if self.current_history_index >= 0:
            prev_state = self.history[self.current_history_index]
        else:
            prev_state = {
                'point_labels': {},
                'region_labels': [],
                'temp_region_points': []
            }
        
        self.point_labels = prev_state['point_labels']
        self.region_labels = [r.copy() for r in prev_state['region_labels']]
        self.temp_region_points = prev_state['temp_region_points']
        
        self.update_visualization()
        print("Undo successful")

    def toggle_transparency(self):
        """Toggle mesh opacity."""
        if self.mesh_opacity == 0.6:
            self.mesh_opacity = 1.0
            print("网格模型设为不透明")
        else:
            self.mesh_opacity = 0.6
            print("网格模型设为半透明")
        
        camera_position = self.plotter.camera_position
        self.update_visualization()
        self.plotter.camera_position = camera_position
        self.plotter.render()

    def update_ply_files_list(self, current_file_path):
        """Refresh the list of PLY files in the current directory."""
        if current_file_path:
            directory = os.path.dirname(current_file_path)
            self.ply_files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.ply')])
            current_file = os.path.basename(current_file_path)
            self.current_file_index = self.ply_files.index(current_file) if current_file in self.ply_files else -1
        else:
            self.ply_files = []
            self.current_file_index = -1

    def load_next_file(self):
        """Load the next PLY file in the same directory."""
        if not self.ply_files:
            print("No PLY files found in current directory")
            return
            
        next_index = (self.current_file_index + 1) % len(self.ply_files)
        
        current_dir = os.path.dirname(self.mesh_file_path) if self.mesh_file_path else os.getcwd()
        next_file = os.path.join(current_dir, self.ply_files[next_index])
        
        try:
            self.point_labels = {}
            self.region_labels = []
            self.temp_region_points = []
            self.selecting_region = False
            self.history = []
            self.current_history_index = -1
            
            if self.load_mesh_file(next_file):
                self.current_file_index = next_index
                self.mesh_file_path = next_file
                print(f"Loaded next file: {self.ply_files[next_index]}")
                
                self.plotter.show(auto_close=False)
        except Exception as e:
            print(f"Error loading next file: {str(e)}")

    def exit_program(self):
        """Exit the program."""
        print("正在退出程序...")
        self.plotter.close()
        sys.exit(0)

# Main program entry
if __name__ == "__main__":
    try:
        labeler = BasicMeshLabeler()
        labeler.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
