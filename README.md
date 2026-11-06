# Brain MRI Segmentation, 3D Reconstruction and FEM

Reproducible research scaffold for the Medical Segmentation Decathlon
`Task01_BrainTumour` dataset. The pipeline is:

`NIfTI -> preprocessing -> U-Net -> marching cubes -> tetrahedral mesh -> linear elasticity FEM`.

The repository contains no patient data. Download it locally with
`python scripts/download_data.py --output-dir data/raw` after reviewing the
dataset license and citation in [data/README.md](data/README.md).

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
python -m brain_mri_fem --help
python -m brain_mri_fem demo --output-dir results/demo
```

The demo uses a synthetic spherical lesion and exercises preprocessing,
surface extraction, metrics, and the FEM solver without downloading data.
Real experiments are configured through the YAML files in `configs/`.

## Current executable baseline

The dependency-light, synthetic end-to-end baseline is executable now:

```powershell
python -m brain_mri_fem demo --output-dir results/demo
```

The preprocessing, U-Net, surface, and FEM components are also importable
individually. Training orchestration, Gmsh integration, and full-dataset
report generation remain the next research phase; their configuration files
are included so experiments use explicit parameters rather than hidden
defaults.

The FEM implementation solves the weak form of linear isotropic elasticity,
`find u in V: a(u,v)=l(v) for every v in V`, with
`a(u,v)=integral epsilon(v):C:epsilon(u) dx`. A fixed boundary and a
configurable traction are applied; results include displacements and von
Mises stress. This is a research baseline, not a clinical tool.

## Layout

* `src/brain_mri_fem`: importable preprocessing, segmentation, reconstruction,
  meshing, FEM, and utility modules.
* `configs`: explicit YAML configuration for every phase.
* `tests`: deterministic unit tests that do not require patient data.
* `notebooks`: Colab-oriented quickstart placeholder and phase notebooks.
* `report`: methodology and reproducibility notes.
* `data`: download instructions only.

Use `python scripts/download_data.py --help` for dataset acquisition. Do not
commit `data/`, checkpoints, NIfTI files, or model weights.
