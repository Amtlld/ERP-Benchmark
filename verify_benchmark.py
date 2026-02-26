import os
import numpy as np
import scipy.io

# 1. Create a dummy .mat file for testing
mat_dir = "test_mat"
os.makedirs(mat_dir, exist_ok=True)

# bpdata shape: [9, 1200, 40, 6]
dummy_bpdata = np.random.rand(9, 1200, 40, 6)
scipy.io.savemat(os.path.join(mat_dir, "S1.mat"), {"bpdata": dummy_bpdata})
scipy.io.savemat(os.path.join(mat_dir, "S2.mat"), {"bpdata": dummy_bpdata})

print("Generated dummy .mat files for testing.")

# 2. Run the preprocessing script
from data_preprocessing.Benchmark.Benchmark_preprocessing import convert_mat_to_npy
npy_dir = "dataset/200Hz/Benchmark"
convert_mat_to_npy(mat_dir, npy_dir)

# 3. Test the MultiDatasetsLoader
import torch
from data_provider.data_loader import MultiDatasetsLoader

class Args:
    pass

args = Args()
args.no_normalize = False
args.root_path = "dataset/200Hz/"  # the root path expects dataset folders inside
args.testing_datasets = "Benchmark"
args.cross_val = "fixed"

loader = MultiDatasetsLoader(args, root_path=args.root_path, flag="TEST")
print(f"Dataset length: {len(loader)}")

features, labels = loader[0]
print(f"Feature shape: {features.shape} (Expected: 1200, 9)")
print(f"Label shape: {labels.shape} (Expected: 4) - Task assigns global IDs")
print(f"Labels content (first item): {labels}")

# Cleanup test directory
import shutil
shutil.rmtree(mat_dir)
shutil.rmtree(npy_dir)
print("Finished Verification.")
