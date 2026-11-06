import h5py

# Remplacez par le chemin exact de votre fichier
file_path = "data/raw/brats2020-h5/BraTS2020_training_data/content/data/volume_1_slice_0.h5"

with h5py.File(file_path, 'r') as f:
    print("--- Clés / Groupes principaux dans le fichier ---")
    print(list(f.keys()))
    
    # Parcourir les datasets pour voir leurs dimensions et types
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} | Forme (Shape): {obj.shape} | Type: {obj.dtype}")

    f.visititems(print_attrs)