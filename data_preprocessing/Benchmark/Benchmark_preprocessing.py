import os
import glob
import numpy as np
import scipy.io

def convert_mat_to_npy(mat_folder_path, npy_dataset_path):
    """
    Convert SSVEP Benchmark .mat files to .npy format suitable for the ERP-Benchmark project.
    
    Expected .mat contents:
    bpdata: [channels(9), timestamps(1200), targets(40), trials(6)]
    
    Expected .npy outputs:
    feature_ID.npy: [trials_total(240), timestamps(1200), channels(9)]
    label_ID.npy: [trials_total(240), labels(3)] -> [target_id, subject_id, trial_id]
    """
    feature_path = os.path.join(npy_dataset_path, "Feature")
    label_path = os.path.join(npy_dataset_path, "Label")
    
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    
    mat_files = glob.glob(os.path.join(mat_folder_path, '*.mat'))
    mat_files.sort()
    
    if not mat_files:
        print(f"No .mat files found in {mat_folder_path}")
        return
        
    print(f"Found {len(mat_files)} .mat files. Starting conversion...")
    
    for file_path in mat_files:
        file_name = os.path.basename(file_path)
        # Extract subject ID from filename, assuming format like 'S1.mat' or similar
        # For a robust approach, we loop through and assign sequential IDs or parse the filename
        try:
            # Try to extract numbers from the filename
            subject_id = int(''.join(filter(str.isdigit, file_name)))
        except ValueError:
            print(f"Warning: Could not extract subject ID from {file_name}. Skipping.")
            continue
            
        print(f"Processing Subject ID: {subject_id} ...")
        
        # Load .mat file
        mat_data = scipy.io.loadmat(file_path)
        
        if 'bpdata' not in mat_data:
            print(f"Error: 'bpdata' not found in {file_name}. Skipping.")
            continue
            
        # Shape is [9, 1200, 40, 6] -> [channels, seq_len, num_targets, num_trials]
        bpdata = mat_data['bpdata']
        
        n_channels, seq_len, n_targets, n_trials_per_target = bpdata.shape
        
        # We need to reshape to [N_total_trials, seq_len, channels]
        # First transpose to [num_targets, num_trials, seq_len, channels]
        # np.transpose(bpdata, (2, 3, 1, 0)) brings shape to (40, 6, 1200, 9)
        reshaped_data = np.transpose(bpdata, (2, 3, 1, 0))
        
        # Then flatten the first two dimensions
        n_total_trials = n_targets * n_trials_per_target
        X = reshaped_data.reshape(n_total_trials, seq_len, n_channels)
        
        # Create labels [n_total_trials, 3]
        y = np.zeros((n_total_trials, 3))
        
        idx = 0
        for target_id in range(n_targets):
            for trial_id in range(n_trials_per_target):
                # We store 0-based target_id (0~39)
                # subject_id -> as extracted, typically (1~35)
                # trial_id -> 1~6 to match your request
                y[idx, 0] = target_id
                y[idx, 1] = subject_id
                y[idx, 2] = trial_id + 1
                idx += 1
                
        # Save to .npy
        # e.g., feature_001.npy, label_001.npy
        feature_filename = os.path.join(feature_path, f"feature_{subject_id:03d}.npy")
        label_filename = os.path.join(label_path, f"label_{subject_id:03d}.npy")
        
        np.save(feature_filename, X)
        np.save(label_filename, y)
        
    print("Conversion completed successfully!")
    print(f"Features saved in: {feature_path}")
    print(f"Labels saved in: {label_path}")

if __name__ == "__main__":
    # Specify the directory where the user stores the .mat files.
    # We'll default to the place we created, or let user decide.
    import argparse
    parser = argparse.ArgumentParser(description="Convert SSVEP .mat to .npy")
    parser.add_argument("--mat_folder", type=str, required=True, help="Path to folder containing .mat files (e.g. /path/to/raw_mat)")
    parser.add_argument("--npy_folder", type=str, default="./dataset/200Hz/Benchmark/", help="Path to save the generated npy dataset")
    
    args = parser.parse_args()
    convert_mat_to_npy(args.mat_folder, args.npy_folder)
