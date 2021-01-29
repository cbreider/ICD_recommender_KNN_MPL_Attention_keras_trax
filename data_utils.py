import pandas as pd
import math
import numpy as np
import gc
import random
from tensorflow import keras


def read_data(file_path, split_ratio=0.0):
    df_train = pd.read_csv(file_path)
    train_icd = [case for case in list(df_train['icd'].str.split(',')) if isinstance(case, list) and len(case) > 1]
    icd_distict = list(set([item for sublist in train_icd for item in sublist]))
    icd_distict.sort()
    train_icd_idx_list = [[icd_distict.index(icd) for icd in case] for case in train_icd]
    random.shuffle(train_icd_idx_list)
    train_data = train_icd_idx_list
    val_data = None
    if split_ratio > 0.0:
        index = math.floor(split_ratio * len(train_data))
        train_data_tmp = train_data
        train_data = train_data_tmp[:index]
        val_data = train_data_tmp[index:]

    del df_train, train_icd, train_icd_idx_list
    gc.collect()

    return train_data, val_data, icd_distict

def read_train_and_val_data(file_path_train, file_path_val):
    df_train = pd.read_csv(file_path_train)
    df_val = pd.read_csv(file_path_val)
    train_icd = [case for case in list(df_train['icd'].str.split(',')) if isinstance(case, list) and len(case) > 1]
    val_icd = [case for case in list(df_val['icd'].str.split(',')) if isinstance(case, list) and len(case) > 1]
    icd_distict = list(set([item for sublist in train_icd for item in sublist]+[item for sublist in val_icd for item in sublist]))
    icd_distict.sort()
    train_icd_idx_list = [[icd_distict.index(icd) for icd in case] for case in train_icd]
    val_icd_idx_list = [[icd_distict.index(icd) for icd in case] for case in val_icd]
    random.shuffle(train_icd_idx_list)
   
    del df_train, train_icd
    gc.collect()

    return train_icd_idx_list, val_icd_idx_list, icd_distict


def to_one_hot(case, n_values):
    a = np.array(case)
    b = np.zeros((a.size, n_values))
    b[np.arange(a.size), a] = 1
    #train_one_hot = np.eye(n_values)[case]
    n_icd = b.shape[0]
    #y_idx = n_icd-1
    y_idx = random.randint(0, n_icd-1)
    y = b[y_idx, :]
    x = np.delete(b, y_idx, 0)
    train_vector_x = np.sum(x, axis=0)# / np.sum(x)

    return train_vector_x, y



