import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
    load_data_by_ids,
)
import warnings
import random

warnings.filterwarnings('ignore')

def get_id_list_benchmark(args, label_path, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for Benchmark data
    Args:
        args: arguments
        label_path: directory of label files
        a: ratio of ids in training set
        b: ratio of ids in training and validation set
    Returns:
        all_ids: list of all IDs
        train_ids: list of IDs for training set
        val_ids: list of IDs for validation set
        test_ids: list of IDs for test set
    '''
    # Extract IDs directly from the filenames or the matrix
    # Based on the preprocessing, expected name is label_{ID}.npy
    all_ids = []
    for filename in os.listdir(label_path):
        if filename.startswith("label_") and filename.endswith(".npy"):
            try:
                subject_id = int(filename.split("_")[1].split(".")[0])
                all_ids.append(subject_id)
            except ValueError:
                pass

    if args.cross_val == 'fixed' or args.cross_val == 'mccv':
        if args.cross_val == 'fixed':
            random.seed(42)  # fixed seed for fixed split
        else:
            random.seed(args.seed)  # random seed for Monte Carlo cross-validation

        all_list = sorted(all_ids)
        random.shuffle(all_list)

        train_ids = all_list[:int(a * len(all_list))]
        val_ids = all_list[int(a * len(all_list)):int(b * len(all_list))]
        test_ids = all_list[int(b * len(all_list)):]

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':  # leave-one-subject-out cross-validation
        # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
        all_list = sorted(all_ids)
        test_ids = [all_list[(args.seed - 41) % len(all_list)]]
        train_ids = [id for id in all_list if id not in test_ids]
        # randomly take 10% of the training set as validation set
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')


class BenchmarkLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/')

        a, b = 0.6, 0.8
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_benchmark(args, self.label_path, a, b)

        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or PRETRAIN.')

        self.X, self.y = load_data_by_ids(self.data_path, self.label_path, ids, args)
        
        # apply normalization as required by the pipeline
        self.X = normalize_batch_ts(self.X)

        # Slice the label to keep target_id, subject_id, trial_id
        # Recall preprocessing generated labels as [target_id, subject_id, trial_id]
        self.y = self.y[:, [0, 1, 2]]

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).float(), \
               torch.from_numpy(np.asarray(self.y[index])).float()

    def __len__(self):
        return len(self.y)
