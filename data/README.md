# Dataset

Use the Medical Segmentation Decathlon Task01_BrainTumour dataset:
<http://medicaldecathlon.com/>. It contains multimodal NIfTI volumes and
labels for whole-tumour segmentation. The dataset is distributed under
CC-BY-SA 4.0; cite the Medical Segmentation Decathlon paper and retain the
license when sharing derived results.

Download locally:

```powershell
python scripts/download_data.py --output-dir data/raw
```

The script downloads only the public archive and extracts it locally. Patient
data must never be committed.
