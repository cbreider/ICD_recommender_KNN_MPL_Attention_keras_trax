import os
import time
import gc
import argparse

# data science imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
import data_utils as du
# utils import


class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with
    KNN implmented by sklearn
    """
    def __init__(self, train_file, test_file, do_five_fold_cs):

        self.train_file = train_file
        self.test_file = test_file
        self.do_five_fold_cs = do_five_fold_cs
        self.model = NearestNeighbors()


    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):

        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def make_recommendations(self, n_recommendations):

        # get data
        data_train, data_test, hashmap, _ = du.read_train_and_val_data_to_index(self.train_file, self.test_file)
        train_data_one_hot = du.to_one_hot_train(data_train, len(hashmap))
        self.model.fit(train_data_one_hot)

        test_one_hot_gen = du.to_one_hot_with_gt_generator(data_test, len(hashmap), False)
        correct = 0
        for i in range(len(data_test)):
            test_case, gt = next(test_one_hot_gen)
            test_case =test_case.reshape((1, -1))
            distances, indices = self.model.kneighbors(test_case, n_neighbors=100)
            distances = distances.flatten()
            indices = indices.flatten()
            test_case = test_case.flatten().astype(np.float)
            icd_pred = []
            case_pred = np.zeros_like(test_case)
            for j, idx in enumerate(indices):
                case_pred += ((train_data_one_hot[idx, :].astype(np.float) - test_case) / distances[j])

            pred_idx = case_pred.argsort()[-5:][::-1]
            for idx in pred_idx:
                icd_pred.append(hashmap[idx])
            gt_idx = np.array(np.where(gt == 1)).item(0)
            gt_icd = hashmap[gt_idx]
            c = False
            if gt_icd in icd_pred:
                c = True
                correct += 1
            print(str(i) + " Predicted: " + str(icd_pred) + "   GT: " + gt_icd + "   " + str(c) + "   " + str(float(correct)/float(i+1)))

        print("Top 5 Acc: " + str(float(correct)/float(len(data_test))))





def parse_args():
    parser = argparse.ArgumentParser(
        prog="ICD Recommender",
        description="Run KNN ICD Recommender")
    parser.add_argument('--train_file', nargs='?', default='train.csv',
                        help='path to train data file')
    parser.add_argument('--test_file', nargs='?', default='test.csv',
                        help='path to test data file')
    parser.add_argument('--do_five_fold', type=bool, default=False,
                        help='path to test data file')
    parser.add_argument('--top_n', type=int, default=5,
                        help='top n icd recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    train_data_path = args.train_file
    test_data_path = args.test_file
    is_five_fold_cross_validation = args.do_five_fold
    top_n = args.top_n

    # initial recommender system
    recommender = KnnRecommender(train_data_path, test_data_path, do_five_fold_cs=is_five_fold_cross_validation)
    # set params
    recommender.set_model_params(20, 'brute', 'cosine', 6)
    # make recommendations
    recommender.make_recommendations(top_n)
