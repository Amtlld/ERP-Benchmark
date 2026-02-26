# BIOT
docker compose exec app python -u run.py --method BIOT --checkpoints_path ./checkpoints/BIOT/pretrain_biot/BIOT/EEG-PREST-16-channels.ckpt \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model BIOT --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CBraMod
docker compose exec app python -u run.py --method CBraMod --checkpoints_path ./checkpoints/CBraMod/pretrain_cbramod/CBraMod/pretrained_weights.pth \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model CBraMod --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# EEGConformer
docker compose exec app python -u run.py --method EEGConformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model EEGConformer --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# EEGFeatures
docker compose exec app python -u run.py --method EEGFeatures \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model EEGFeatures --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# EEGInception
docker compose exec app python -u run.py --method EEGInception \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model EEGInception --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# EEGNet
docker compose exec app python -u run.py --method EEGNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model EEGNet --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ERPFeatures
docker compose exec app python -u run.py --method ERPFeatures \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model ERPFeatures --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# iTransformer
docker compose exec app python -u run.py --method iTransformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model iTransformer --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# LaBraM
docker compose exec app python -u run.py --method LaBraM --checkpoints_path ./checkpoints/LaBraM/pretrain_labram/LaBraM/labram-base.pth \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model LaBraM --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 128 --swa --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# Medformer
docker compose exec app python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model Medformer --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# MedGNN
docker compose exec app python -u run.py --method MedGNN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model MedGNN --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --resolution_list 2,4,6,8 --nodedim 10 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ModernTCN
docker compose exec app python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model ModernTCN --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PatchTST
docker compose exec app python -u run.py --method PatchTST \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model PatchTST --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 64 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TCN
docker compose exec app python -u run.py --method TCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model TCN --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TestFormer - Multi-Variate
docker compose exec app python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model TestFormer --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TestFormer - Uni-Variate
docker compose exec app python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model TestFormer --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 100 --patch_type uni-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TestFormer - Whole-Variate
docker compose exec app python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model TestFormer --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_type whole-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# TimesNet
docker compose exec app python -u run.py --method TimesNet \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-Benchmark --model TimesNet --data MultiDatasets \
--training_datasets Benchmark \
--testing_datasets Benchmark \
--e_layers 2 --batch_size 128 --top_k 1 --d_model 32 --d_ff 64 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15