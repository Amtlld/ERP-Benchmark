import os
import glob

# The directory containing all the script folders
scripts_dir = "/home/ubuntu/dl_projects/ERP-Benchmark/scripts"

# Find all S-1.sh scripts
sh_files = glob.glob(os.path.join(scripts_dir, "*", "supervised", "*", "S-1.sh"))
sh_files.extend(glob.glob(os.path.join(scripts_dir, "*", "S-1.sh"))) # some might not have 'supervised' in path

print(f"Found {len(sh_files)} script files.")

# The template to append
template = """
# Benchmark
python -u run.py --method {method} \\
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model {model} --data MultiDatasets \\
--training_datasets Benchmark \\
--testing_datasets Benchmark \\
--batch_size 128 --swa \\
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
"""

for file_path in set(sh_files):
    # Extract method and model from the file path
    # Example path: /home/.../scripts/EEGNet/supervised/EEGNet/S-1.sh
    parts = file_path.split(os.sep)
    
    # We can usually infer the model and method name from the parent components
    try:
        # The parent directory of S-1.sh is usually the model name
        model_name = parts[-2]
        # The topmost directory under scripts/ is usually the method name
        method_name = parts[parts.index("scripts") + 1]
    except Exception as e:
        print(f"Skipping {file_path} due to parsing error: {e}")
        continue
    
    # Render the command
    command_to_append = template.format(method=method_name, model=model_name)
    
    # Append to the file
    with open(file_path, "a") as f:
        f.write(command_to_append)
        
    print(f"Updated {file_path} with model={model_name} and method={method_name}")

print("Update completed.")
