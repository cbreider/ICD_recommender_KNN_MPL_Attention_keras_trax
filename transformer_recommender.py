import trax.layers as tl
import trax.supervised as ts
import trax
from datetime import datetime
import argparse
import os
import data_utils as dt
import math


def RecommenderTransformer(n_classes_in, embedding_size, n_out_classes, dropout_rate):
    transfomer = tl.Serial(
        tl.Embedding(n_classes_in, d_feature=embedding_size),
        tl.Flatten(),
        tl.Dropout(dropout_rate),
        tl.SelfAttention(4),
        tl.Dropout(dropout_rate),
        #tl.DotProductCausalAttention(4),
        tl.Dense(n_out_classes),
        tl.LogSoftmax()
    )

    print(str(transfomer))
    return transfomer


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ICD Recommender",
        description="Run MLP ICD Recommender")
    parser.add_argument('--train_file', nargs='?', default='train.csv',
                        help='path to train data file')
    parser.add_argument('--test_file', nargs='?', default='test.csv',
                        help='path to test data file')
    parser.add_argument('--out_folder', nargs='?', default=None,
                        help='path for output files')
    parser.add_argument('--model_path', nargs='?', default=None,
                        help='path to import model and just run inference on test data')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.05,
                        help='dropout rate')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='size of embedding_layer, choose 0 for no embedding layer')
    parser.add_argument('--do_five_fold', type=bool, default=False,
                        help='TODO')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    train_data_path = args.train_file
    test_data_path = args.test_file
    out_folder = args.out_folder
    model_path = args.model_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    embedding_size = args.embedding_size
    is_five_fold_cross_validation = args.do_five_fold  # TODO

    if out_folder is None:
        out_folder = "model_output/RUN_{}".format(datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    text_file = open("{}/params.txt".format(out_folder), "w")
    n = text_file.write(str(args))
    text_file.close()

    #train_data, val_data, classes, len_input = dt.read_data_to_index(train_data_path, min_length=1, split_ratio=0.9)
    train_data, val_data, classes, len_input = dt.read_train_and_val_data_to_index(train_data_path, test_data_path)

    recommender = RecommenderTransformer(n_classes_in=len(classes),
                                         embedding_size=embedding_size,
                                         n_out_classes=len(classes),
                                         dropout_rate=dropout_rate)

    n_train_b = math.floor(float(len(train_data)) / float(batch_size))
    train_task = ts.training.TrainTask(
        labeled_data=dt.get_input_sequence_and_gt(train_data, len_input, batch_size),
        loss_layer=tl.CategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(learning_rate),
        n_steps_per_checkpoint=400, #This will print the results at every 200 training steps.
    )
    n_eval_b = math.floor(float(len(val_data)) / float(batch_size))
    # Evaluaton task.
    eval_task = ts.training.EvalTask(
        labeled_data=dt.get_input_sequence_and_gt(val_data, len_input, batch_size),
        metrics=[tl.CategoryCrossEntropy(), tl.CategoryAccuracy()],
        n_eval_batches=n_eval_b)

    training_loop = ts.training.Loop(recommender,
                                     train_task,
                                     eval_tasks=[eval_task],
                                     output_dir=out_folder)

    # Run 2000 steps (batches).
    training_loop.run(2000)

