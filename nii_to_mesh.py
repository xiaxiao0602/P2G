import nibabel as nib
import numpy as np
import pymeshlab
import os
from skimage import measure
from scipy import ndimage


def extract_and_convert_to_mesh(nii_file, label_value, smoothing_iterations):
    """
    Extract a label from a NIfTI mask, convert to mesh, and apply smoothing.

    Returns
    -------
    pymeshlab.MeshSet | None
    """
    try:
        print(f"正在读取文件: {nii_file}")
        nii_img = nib.load(nii_file)
        volume_data = nii_img.get_fdata()
        
        print(f"正在提取标签 {label_value} 的区域")
        mask = (volume_data == label_value)
        
        if not np.any(mask):
            print(f"标签 {label_value} 在文件中不存在，跳过处理")
            return None
        
        z_dim, y_dim, x_dim = mask.shape
        on_boundary = False
        
        if np.any(mask[0, :, :]) or np.any(mask[z_dim-1, :, :]) or \
           np.any(mask[:, 0, :]) or np.any(mask[:, y_dim-1, :]) or \
           np.any(mask[:, :, 0]) or np.any(mask[:, :, x_dim-1]):
            print(f"标签 {label_value} 的区域位于边界面上，跳过处理")
            on_boundary = True
            return None
        
        if on_boundary:
            return None
        
        print("正在对体素数据进行平滑...")
        mask_float = mask.astype(float)
        mask_smooth = ndimage.gaussian_filter(mask_float, sigma=1.8)
        
        max_val = np.max(mask_smooth)
        min_val = np.min(mask_smooth)
        if max_val - min_val < 0.1:
            print(f"警告: 标签 {label_value} 的区域平滑后值域太小 ({min_val:.3f}-{max_val:.3f})，无法生成网格")
            return None
        
        print("正在生成mesh...")
        try:
            level = min_val + (max_val - min_val) * 0.3
            verts, faces, normals, values = measure.marching_cubes(
                mask_smooth,
                level=level,
                spacing=nii_img.header.get_zooms(),
                step_size=1,
                allow_degenerate=False
            )
        except ValueError as e:
            if "Surface level must be within volume data range" in str(e):
                print(f"警告: 标签 {label_value} 无法使用marching cubes生成网格: {str(e)}")
                print(f"数据范围: {min_val:.3f}-{max_val:.3f}, 尝试使用最小值作为阈值")
                try:
                    verts, faces, normals, values = measure.marching_cubes(
                        mask_smooth,
                        level=min_val + 0.001,
                        spacing=nii_img.header.get_zooms(),
                        step_size=1,
                        allow_degenerate=False
                    )
                except Exception as e2:
                    print(f"错误: 无法生成标签 {label_value} 的网格: {str(e2)}")
                    return None
            else:
                print(f"错误: 生成标签 {label_value} 的网格时出错: {str(e)}")
                return None
        except Exception as e:
            print(f"错误: 生成标签 {label_value} 的网格时出错: {str(e)}")
            return None
        
        if len(verts) < 3 or len(faces) < 1:
            print(f"警告: 标签 {label_value} 生成的网格无效，顶点数: {len(verts)}, 面数: {len(faces)}")
            return None
        
        ms = pymeshlab.MeshSet()
        
        mesh = pymeshlab.Mesh(verts, faces)
        ms.add_mesh(mesh)
        
        print(f"正在进行mesh平滑处理...")
        
        try:
            print("步骤1: Taubin平滑...")
            for _ in range(3):  # 增加Taubin平滑的轮数
                ms.apply_filter('apply_coord_taubin_smoothing', 
                                lambda_=0.5,
                                mu=-0.53,
                                stepsmoothnum=smoothing_iterations
                )
            
            print("步骤2: 拉普拉斯平滑...")
            for _ in range(2):  # 多次应用拉普拉斯平滑
                ms.apply_filter('apply_coord_laplacian_smoothing', 
                                stepsmoothnum=smoothing_iterations,
                                cotangentweight=True,
                                boundary=True
                )
            
            print("步骤3: 最终Taubin平滑...")
            ms.apply_filter('apply_coord_taubin_smoothing', 
                            lambda_=0.3,
                            mu=-0.32,
                            stepsmoothnum=smoothing_iterations
            )
        except Exception as e:
            print(f"警告: 平滑标签 {label_value} 的网格时出错: {str(e)}")
        
        return ms
    except Exception as e:
        print(f"处理标签 {label_value} 时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_subfolder(subfolder, label_values, output_dir, smoothing_iterations, output_format='stl'):
    """
    Process one case folder that contains mask.nii.gz.
    """
    try:
        mask_path = os.path.join(subfolder, "mask.nii.gz")
        if not os.path.exists(mask_path):
            print(f"在 {subfolder} 中未找到 mask.nii.gz 文件，跳过")
            return
        
        subfolder_name = os.path.basename(subfolder)
        
        for label_value in label_values:
            try:
                print(f"\n处理 {subfolder_name} 中的标签 {label_value}")
                
                output_filename = f"{label_value}_{subfolder_name}.{output_format}"
                output_path = os.path.join(output_dir, output_filename)
                
                if os.path.exists(output_path):
                    print(f"文件 {output_filename} 已存在，跳过处理")
                    continue
                
                ms = extract_and_convert_to_mesh(mask_path, label_value, smoothing_iterations)
                
                if ms is not None:
                    print(f"正在保存mesh到文件: {output_path}")
                    ms.save_current_mesh(output_path)
                    print(f"标签 {label_value} 处理完成!")
                else:
                    print(f"标签 {label_value} 处理失败或不满足提取条件，跳过")
            except Exception as e:
                print(f"处理标签 {label_value} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                print(f"继续处理下一个标签...")
                continue
    except Exception as e:
        print(f"处理子文件夹 {subfolder} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    input_dir = os.path.join("data", "nifti_cases")
    output_dir = os.path.join("outputs", "meshes_from_nii")
    label_values = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    smoothing_iterations = 3
    output_format = 'ply'
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        subfolders = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir) 
                      if os.path.isdir(os.path.join(input_dir, folder))]
    except Exception as e:
        print(f"获取子文件夹列表时出错: {str(e)}")
        return
    
    if not subfolders:
        print(f"在 {input_dir} 中未找到子文件夹")
        return
    
    if output_format.lower() not in ['stl', 'ply']:
        print(f"输出格式 {output_format} 无效，必须是 'stl' 或 'ply'，默认使用 'stl'")
        output_format = 'stl'
    
    total_folders = len(subfolders)
    processed_folders = 0
    failed_folders = 0
    failed_list = []
    
    for i, subfolder in enumerate(subfolders):
        try:
            folder_name = os.path.basename(subfolder)
            print(f"\n[{i+1}/{total_folders}] 开始处理子文件夹: {folder_name}")
            process_subfolder(subfolder, label_values, output_dir, smoothing_iterations, output_format)
            processed_folders += 1
            print(f"子文件夹 {folder_name} 处理完成")
        except Exception as e:
            print(f"处理子文件夹 {subfolder} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_folders += 1
            failed_list.append(folder_name)
            print(f"继续处理下一个子文件夹...")
            continue
    
    print("\n========= 处理结果摘要 =========")
    print(f"总文件夹数: {total_folders}")
    print(f"成功处理: {processed_folders}")
    print(f"处理失败: {failed_folders}")
    
    if failed_list:
        print("\n失败的文件夹列表:")
        for folder in failed_list:
            print(f"- {folder}")
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    main() 
