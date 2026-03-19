import os
import json
import numpy as np
import pyvista as pv

def create_training_data(ply_path, output_dir):
    """
    Convert a labeled PLY (+ JSON annotations) into a TXT point cloud for training.

    Output format per line:
    x y z nx ny nz label
    """
    json_path = ply_path.replace('.ply', '.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing annotation JSON for PLY: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if len(data['point_labels']) != 2:
        raise ValueError(f"关键点数量不是2，当前有{len(data['point_labels'])}个关键点")
    
    if len(data['region_labels']) != 2:
        raise ValueError(f"区域数量不是2，当前有{len(data['region_labels'])}个区域")
    
    mesh = pv.read(ply_path)
    points = mesh.points
    normals = mesh.point_normals
    
    labels = np.zeros(len(points), dtype=np.int32)
    
    for point_idx in data['point_labels'].keys():
        labels[int(point_idx)] = 1
    
    for region in data['region_labels']:
        for point_idx in region['points']:
            labels[point_idx] = 2
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, os.path.basename(ply_path).replace('.ply', '.txt'))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(points)):
            x, y, z = points[i]
            nx, ny, nz = normals[i]
            label = labels[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {label}\n")
    
    print(f"已生成训练数据文件: {output_path}")

if __name__ == '__main__':
    ply_dir = os.path.join("data", "training", "ply_labeled")
    output_dir = os.path.join("outputs", "training_txt")
    
    if not os.path.isdir(ply_dir):
        raise FileNotFoundError(
            f"Missing directory: {ply_dir}. Put labeled PLY+JSON files there or update the paths in __main__."
        )

    for filename in os.listdir(ply_dir):
        if filename.endswith('.ply'):
            ply_path = os.path.join(ply_dir, filename)
            try:
                create_training_data(ply_path, output_dir)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}") 
