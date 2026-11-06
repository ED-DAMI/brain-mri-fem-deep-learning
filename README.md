# Brain MRI Segmentation and 3D Reconstruction using Deep Learning and FEM

**A hybrid approach combining deep learning and the Finite Element Method (FEM) for brain MRI analysis and 3D modeling.**

> **Project status:** In development — this repository documents our planned methodology. Implementation and experimental results will be added progressively.

## Overview

This academic project explores a hybrid workflow for processing **brain magnetic resonance imaging (MRI)** data. Deep learning will be used to segment anatomical structures or regions of interest, while the **Finite Element Method (FEM)** will be used on a mesh derived from the segmentation for mathematical modeling and, potentially, deformation or regularization.

Our goal is to connect **medical image analysis**, **3D reconstruction**, and **numerical simulation** within a reproducible Python-based pipeline.

## Objectives

- Preprocess 2D/3D brain MRI data (normalization, resizing, and data preparation).
- Train and evaluate a deep learning model for brain MRI segmentation.
- Reconstruct a 3D surface from the predicted segmentation masks.
- Generate a mesh suitable for finite element analysis.
- Apply an FEM-based numerical model to the reconstructed geometry.
- Visualize and evaluate the results at each stage.

## Proposed Workflow

```text
Brain MRI scans
      |
      v
Preprocessing and normalization
      |
      v
Deep learning segmentation (U-Net baseline)
      |
      v
3D reconstruction of segmented regions
      |
      v
Surface / volume mesh generation
      |
      v
Finite element modeling (FEM)
      |
      v
3D visualization and evaluation
```

## Methodology

### 1. Brain MRI preprocessing

Load MRI scans, inspect image dimensions and voxel spacing, normalize intensity values, and prepare training/validation/test splits.

### 2. Deep learning segmentation

Start with a **U-Net** model to segment a selected brain structure or region of interest. Evaluate segmentation quality using metrics such as the **Dice coefficient** and **Intersection over Union (IoU)**.

### 3. 3D reconstruction

Combine segmentation predictions into a 3D volume, extract a surface (for example, using **Marching Cubes**), and visualize the reconstructed structure.

### 4. Finite Element Method (FEM)

Create a **volumetric mesh** from the reconstructed geometry and formulate a numerical problem on this mesh. A possible first experiment is a simplified **linear-elastic deformation** with specified material parameters and boundary conditions. This ensures that FEM is used to **solve a mathematical model**, rather than only to display a mesh.

### 5. Evaluation

Assess segmentation performance, inspect reconstructed geometries and mesh quality, and visualize the numerical FEM solution. Quantitative results will be added once experiments have been completed.

## Proposed Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Medical imaging:** NiBabel and/or SimpleITK
- **Image processing and 3D reconstruction:** NumPy, scikit-image
- **Meshing and visualization:** Gmsh, PyVista
- **Finite element analysis:** FEniCSx or another appropriate FEM solver

*The final libraries may change as the implementation progresses.*

## Planned Repository Structure

```text
brain-mri-fem-deep-learning/
├── data/                 # Dataset instructions; no patient data committed
├── notebooks/            # Exploration and experiments
├── src/
│   ├── preprocessing/    # MRI loading and preprocessing
│   ├── segmentation/     # Deep learning models and training
│   ├── reconstruction/   # 3D reconstruction and meshing
│   └── fem/              # Finite element simulations
├── results/              # Figures and experiment summaries
├── requirements.txt      # Dependencies (to be added)
└── README.md
```



## Dataset and Privacy

The recommended dataset is BraTS 2020 from Kaggle:
`awsaf49/brats2020-training-data`. Install the Kaggle client and download it
with:

```powershell
python -m pip install kagglehub
python scripts\download_data.py --source kaggle --output-dir data\raw
```

The script copies the files into `data\raw\brats2020-training-data` and checks
that NIfTI or HDF5 files are present. The dataset may require Kaggle authentication
depending on the account and download policy. **Medical images and sensitive
data will not be committed to this repository.**

The exploratory notebook is [01_eda_brats2020.ipynb](notebooks/01_eda_brats2020.ipynb).
Open it from the repository root and run all cells before preprocessing.

The Kaggle version currently downloaded contains HDF5 2D slices (`image` and
`mask`, shape `240x240`) rather than original 3D NIfTI volumes. Use
[01_eda_brats2020_h5.ipynb](notebooks/01_eda_brats2020_h5.ipynb) for this
download. It is suitable for a 2D U-Net; 3D reconstruction and FEM require
the original 3D BraTS NIfTI dataset.

## Roadmap

- [ ] Select the MRI dataset and segmentation target
- [ ] Implement preprocessing and data exploration
- [ ] Train and evaluate a U-Net baseline
- [ ] Reconstruct segmented regions in 3D
- [ ] Generate and inspect a volumetric mesh
- [ ] Formulate and solve a simplified FEM problem
- [ ] Compare results and prepare the final report

## Team

Academic mini-project developed by **Mohamed Eddami** and **Israe Yaaya**.

## Academic Context

This repository is part of a student project on the **Finite Element Method**, with an application to **brain MRI segmentation and 3D reconstruction**.
