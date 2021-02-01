import pandas as pd
import math
import numpy as np
import gc
import random


def read_icd_data(file_path):
    df = pd.read_csv(file_path)
    icd = [case for case in list(df['icd'].str.split(',')) if isinstance(case, list)]
    del df
    gc.collect()
    return icd


def get_icd_distinct(data):
    data_distinct = list(set([item for sublist in data for item in sublist]))
    data_distinct.sort()
    return data_distinct


def get_icd_idx(data, lookup):
    icd_idx_list = [[lookup.index(icd) for icd in case] for case in data]
    return icd_idx_list


def filter_by_length(data, min_length=0):
    return [case for case in data if len(case) > min_length]


def read_data_to_index(file_path, min_length=0, split_ratio=0.0):
    icd_data = read_icd_data(file_path)
    icd_distict = get_icd_distinct(icd_data)
    icd_data = filter_by_length(icd_data, min_length=min_length)
    icd_idx_list = get_icd_idx(icd_data, icd_distict)
    n_classes = max([len(c) for c in icd_idx_list])
    random.shuffle(icd_idx_list)
    train_data = icd_idx_list
    val_data = None
    if split_ratio > 0.0:
        index = math.floor(split_ratio * len(train_data))
        train_data_tmp = train_data
        train_data = train_data_tmp[:index]
        val_data = train_data_tmp[index:]

    del icd_data, icd_idx_list
    gc.collect()

    return train_data, val_data, icd_distict, n_classes


def read_train_and_val_data_to_index(file_path_train, file_path_val):
    train_icd = read_icd_data(file_path_train)
    val_icd = read_icd_data(file_path_val)
    icd_distict = get_icd_distinct(train_icd + val_icd)
    train_icd = filter_by_length(train_icd, min_length=1)
    val_icd = filter_by_length(val_icd, min_length=1)
    train_icd_idx_list = get_icd_idx(train_icd, icd_distict)
    val_icd_idx_list = get_icd_idx(val_icd, icd_distict)
    n_classes = max([len(c) for c in train_icd_idx_list + val_icd_idx_list])
    del train_icd, val_icd
    gc.collect()

    return train_icd_idx_list, val_icd_idx_list, icd_distict, n_classes


def read_test_data_for_prediction(file_path, icd_distict, normalize=False):
    df = pd.read_csv(file_path)
    case_id = [case for case in df['case_id']]
    icd = [case for case in list(df['icd'].str.split(',')) if isinstance(case, list)]
    train_icd_idx_list = get_icd_idx(icd, icd_distict)
    nr_val = len(icd_distict)
    case_dict = dict()
    for case in zip(case_id, train_icd_idx_list):
        id = case[0]
        icds = case[1:]
        idcs_oh = to_one_hot(icds, nr_val)
        oh = np.sum(idcs_oh, axis=0)
        if normalize:
            oh /= np.sum(oh)
        case_dict[id] = oh
    del df, case_id, icd, train_icd_idx_list
    gc.collect()

    return case_dict


def to_one_hot(case, n_values):
    np_case = np.array(case)
    one_hot = np.zeros((np_case.size, n_values), np.int32)
    one_hot[np.arange(np_case.size), np_case] = 1
    return one_hot


def get_input_sequence_and_gt(cases, max_len, batch_size):
    random.shuffle(cases)
    count = 0
    while True:
        batch_x = np.zeros((batch_size, max_len), dtype=np.int32)
        batch_y = np.zeros((batch_size,), dtype=np.int32)
        for i in range(0, batch_size):
            case = cases[count]
            case.sort()
            case_np = np.array(case)
            y_idx = random.randint(0, (len(case)-1))
            y_j = case_np[y_idx]
            x_j = np.zeros((max_len), dtype=np.int32)
            x_tmp = np.delete(case, y_idx, 0) + 1
            x_j[:x_tmp.shape[0]] = x_tmp
            batch_x[i,:] = x_j
            batch_y[i] = y_j
            count += 1
            if count == len(cases):
                random.shuffle(cases)
                count = 0
        yield ((batch_x, batch_y))


def to_one_hot_with_gt_generator(cases, n_values, random_gt_index=True, normalize=False):
    random.shuffle(cases)
    avg = 0
    for c in cases:
        avg += len(c)

    avg /= len(cases)
    for case in cases:
        oh = to_one_hot(case, n_values)
        n_icd = oh.shape[0]
        y_idx = n_icd-1
        if random_gt_index:
            y_idx = random.randint(0, n_icd-1)
        y = oh[y_idx, :]
        x = np.delete(oh, y_idx, 0)
        train_vector_x = np.sum(x, axis=0)
        if normalize:
            train_vector_x /= avg
        yield train_vector_x, y



