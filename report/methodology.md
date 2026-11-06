# Methodology

The primary target is binary whole-tumour segmentation (`label > 0`) from
FLAIR. Splits are made by patient before preprocessing to avoid leakage.
Images are normalized on non-zero brain voxels and retain NIfTI affine and
spacing metadata.

The neural baseline is a 3D U-Net trained with a configurable Dice plus
cross-entropy objective. A predicted binary volume is converted to a physical
surface with marching cubes. The FEM baseline uses four-node tetrahedra and
linear isotropic elasticity. For displacement `u` and test function `v`,
`a(u,v)=l(v)` where `a(u,v)=int epsilon(v):C:epsilon(u) dx`; Dirichlet
constraints remove rigid-body modes and a traction contributes to `l`.

All reported experiments must record the config, seed, data split JSON,
software environment, metrics, and failure cases. Results are exploratory and
must not be interpreted as clinical predictions.
