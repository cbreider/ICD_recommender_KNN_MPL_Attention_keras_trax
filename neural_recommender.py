from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, Reshape, Dot, Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add, Activation, Lambda
from tensorflow import data, keras
import argparse
import data_utils as dt
from datetime import datetime
import matplotlib.pyplot as plt
import pickle as pkl
import os
import tensorflow as tf
import  numpy as np


def RecommenderNet(n_in_classes, n_out_classes, n_layers, layer_size, decay_layer_size, dropout_rate, learning_rate,
                   embedding_size=0):
    """model = Sequential()
    model.add(Embedding(n_in_classes, embedding_size, embeddings_initializer='he_normal',
                        embeddings_regularizer=l2(1e-6)))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    for i in range(0, n_layers):
        model.add(Dense(layer_size, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_out_classes, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    opt = Adam(lr=learning_rate)
    """
    input_layer = Input(shape=(n_in_classes))
    x = input_layer

    if embedding_size > 0:
        embeddings = Embedding(n_in_classes, embedding_size, embeddings_initializer='he_normal',
                               embeddings_regularizer=l2(1e-6))(input_layer)
        embeddings_flat = Flatten()(embeddings)
        embeddings_d = Dropout(dropout_rate)(embeddings_flat)
        x = embeddings_d

    for i in range(0, n_layers):
        x = Dense(layer_size, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        if decay_layer_size:
            layer_size = layer_size / 2

    x = Dense(n_out_classes, kernel_initializer='he_normal')(x)
    y = Activation('softmax')(x)
    model = Model(inputs=input_layer, outputs=y)
    opt = Adam(lr=learning_rate)

    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

    return model


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
    parser.add_argument('--nr_hidden_layers', type=int, default=3,
                        help='number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=32,
                        help='size of layers')
    parser.add_argument('--decay_layer_size', type=bool, default=False,
                        help='half layer size for every layer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.05,
                        help='dropout rate')
    parser.add_argument('--embedding_size', type=int, default=0,
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
    nr_hidden_layers = args.nr_hidden_layers
    layer_size = args.layer_size
    decay_layer_size = args.decay_layer_size
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

    #train_data, val_data, classes = dt.read_data_to_index(train_data_path, min_length=1, split_ratio=0.9)
    train_data, val_data, classes = dt.read_train_and_val_data_to_index(train_data_path, test_data_path)

    training_generator = data.Dataset.from_generator(
        lambda : dt.to_one_hot_with_gt_generator(train_data, len(classes), True, False),
        output_types=(tf.int32, tf.int32))
    training_generator = training_generator.shuffle(buffer_size=4096).batch(batch_size)

    validation_generator = data.Dataset.from_generator(
        lambda : dt.to_one_hot_with_gt_generator(val_data, len(classes), True, False),
        output_types=(tf.int32, tf.int32))
    validation_generator = validation_generator.shuffle(buffer_size=4096).batch(1)

    if model_path is not None:
        recommender = keras.models.load_model(model_path)
    else:
        # initial recommender system
        recommender = RecommenderNet(
            n_in_classes=len(classes),
            n_out_classes=len(classes),
            embedding_size=embedding_size,
            n_layers=nr_hidden_layers,
            layer_size=layer_size,
            decay_layer_size=decay_layer_size,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate)

        # set params
        model_hist = recommender.fit(training_generator, validation_data=validation_generator,
                                               epochs=num_epochs, use_multiprocessing=True,
                                               workers=6, validation_freq=5)

        plt.plot(model_hist.history['top_k_categorical_accuracy'])
        plt.plot(model_hist.history['val_top_k_categorical_accuracy'])
        plt.plot(model_hist.history['val_categorical_accuracy'])
        plt.plot(model_hist.history['categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_top_5_acc', 'val_top_5_acc', 'train_acc', 'val_acc'], loc='upper left')
        plt.savefig("{}/acc_hist.jpg".format(out_folder))

        recommender.save(out_folder)
        with open('{}/model_history.pickle'.format(out_folder), 'wb') as handle:
            pkl.dump(model_hist.history, handle)

    result = '"case_id","icd"\n'
    test_data = dt.read_test_data_for_prediction(test_data_path, classes, False)

    for id, icds in test_data.items():
        result = result + '"{}"'.format(id) + ','
        prediction = recommender.predict(np.array([icds,]))[0,:]
        top_five_pred = prediction.argsort()[-5:][::-1]
        icd_string = '"'
        for top in top_five_pred:
            icd_name = classes[top]
            icd_string += icd_name + ','
        icd_string = icd_string[:-1] + '"'
        result += icd_string + '\n'


    text_file = open("{}/prediction.csv".format(out_folder), "w")
    n = text_file.write(result)
    text_file.close()


