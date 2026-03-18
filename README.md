# P2G
P2G (Point to Guide) is an end-to-end framework that integrates point-cloud deep learning with geometric optimization to automatically design 3D-printable patient-specific pedicle screw guides from CT scans.

<img width="403" height="443" alt="Overview" src="https://github.com/user-attachments/assets/d9afc879-b3d8-4eb8-b6a7-ab7942d83758" />

*Figure 1: End-to-end framework of P2G: from CT to 3D-printable patient-specific pedicle screw guides.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Repo stars](https://img.shields.io/github/stars/xiaxiao0602/P2G?style=social)](https://github.com/xiaxiao0602/P2G)

P2G is a **fully automated, end-to-end framework** that integrates multi-task point cloud deep learning with geometric optimization to design patient-specific pedicle screw guides from preoperative CT scans. The framework automatically predicts screw entry points, segments posterior contact regions, computes the safest screw trajectory by maximizing cortical clearance, and generates a 3D-printable STL guide—all in about **3 minutes** per case.

---

## Dataset

We use the publicly available [CTSpine1k](https://github.com/MIRACLE-Center/CTSpine1K) dataset. After automatic segmentation with TotalSegmentator and manual quality control, **5,526 vertebrae** (T7–L5) were retained. Each vertebra is represented as a point cloud with two annotations:

- **Entry Point**: A keypoint at the junction of transverse process and superior articular process.
- **Contact Region**: A binary mask on the posterior arch (lamina, spinous process base, articular processes) ensuring stable guide placement.

The raw point clouds are available in this repository under `data/`.

---

## Method Overview

The pipeline consists of four main stages:

1. **Vertebral Reconstruction**: CT → segmentation ([TotalSegmentator](https://github.com/wasserth/TotalSegmentator)) → mesh → point cloud.
2. **Feature Extraction**: A multi-task point cloud network predicts entry point heatmaps and contact region masks.
3. **Trajectory Optimization**: Candidate screw directions are sampled within a cone; the direction maximizing the minimum distance to cortical bone is selected.
4. **Parametric Guide Generation**: Based on the predicted features and optimal trajectory, a 3D guide is automatically modeled (base, bridge, drill sleeves) and exported as STL.

<img width="733" height="312" alt="image" src="https://github.com/user-attachments/assets/b4db49f3-1c6b-4441-8cd5-b1f0bb5081a4" />

*Figure 2: Multi-task point cloud network architecture.*

For details, please refer to our paper.

---

## Web Application Demo

We have integrated the P2G framework into a **user-friendly online platform**. The platform provides an intuitive interface for uploading CT data, visualizing intermediate results, and downloading the final STL guide.

https://github.com/user-attachments/assets/ebf8325d-fcb6-407f-ab19-cae957ef3189








