# P2G
P2G (Point to Guide) is an end-to-end framework that integrates point-cloud deep learning with geometric optimization to automatically design 3D-printable patient-specific pedicle screw guides from CT scans.
<img width="1614" height="1374" alt="Overview" src="https://github.com/user-attachments/assets/d9afc879-b3d8-4eb8-b6a7-ab7942d83758" />
*Figure 1: End-to-end pipeline of P2G: from CT to 3D-printable patient-specific pedicle screw guides.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Repo stars](https://img.shields.io/github/stars/yyu-lab/P2G?style=social)](https://github.com/yyu-lab/P2G)

P2G is a **fully automated, end-to-end framework** that integrates multi-task point cloud deep learning with geometric optimization to design patient-specific pedicle screw guides from preoperative CT scans. The framework automatically predicts screw entry points, segments posterior contact regions, computes the safest screw trajectory by maximizing cortical clearance, and generates a 3D-printable STL guide—all in about **3 minutes** per case.

---

## 🚀 Key Features

- **End-to-End Automation**: From CT to 3D-printable STL, no manual intervention required.
- **Multi-Task Point Cloud Learning**: Simultaneously predicts entry points (via heatmap regression) and contact regions (via segmentation) using architectures like PointNet++, PointNeXt, and Point Transformer V3.
- **Geometric Trajectory Optimization**: Maximizes minimum distance to cortical bone to ensure safe screw placement.
- **Parametric Guide Generation**: Automatically constructs watertight, manufacturable guide models with integrated drill sleeves and cross-linking bridges.
- **Clinically Validated**: Evaluated on 5,526 vertebrae (T7–L5) from the CTSpine1k dataset and validated ex vivo on 3D-printed models, achieving mean entry-point errors <2.3 mm and angular deviations <2.2°.

---

## 📊 Dataset

We use the publicly available [CTSpine1k](https://github.com/ICTMCG/CTSpine1k) dataset. After automatic segmentation with TotalSegmentator and manual quality control, **5,526 vertebrae** (T7–L5) were retained. Each vertebra is represented as a point cloud (8,192 points) with two annotations:

- **Entry Point**: A keypoint at the junction of transverse process and superior articular process.
- **Contact Region**: A binary mask on the posterior arch (lamina, spinous process base, articular processes) ensuring stable guide placement.

The raw and annotated point clouds are available in this repository under `data/`.

---

## 🧠 Method Overview

![Method Overview](docs/figures/figure3.png)  
*Figure 3: Multi-task point cloud network architecture.*

The pipeline consists of four main stages:

1. **Vertebral Reconstruction**: CT → segmentation (TotalSegmentator) → mesh → point cloud.
2. **Feature Extraction**: A multi-task point cloud network predicts entry point heatmaps and contact region masks.
3. **Trajectory Optimization**: Candidate screw directions are sampled within a cone; the direction maximizing the minimum distance to cortical bone is selected.
4. **Parametric Guide Generation**: Based on the predicted features and optimal trajectory, a 3D guide is automatically modeled (base, bridge, drill sleeves) and exported as STL.

For details, please refer to our paper (see [Citation](#citation)).

---

## ⚙️ Installation & Usage

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)
- Other dependencies: see `requirements.txt`

### Installation
```bash
git clone https://github.com/yyu-lab/P2G.git
cd P2G
pip install -r requirements.txt
