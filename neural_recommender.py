from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add, Activation, Lambda
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import keras
import argparse
import numpy as np
import data_utils as dt


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, classes, batch_size=32, shuffle=True):
        'Initialization'
        self.data = data
        self.classes = classes
        self.n_classes = n_values = len(self.classes) + 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_cases_tmp = [self.data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_cases_tmp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_cases_tmp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_classes))
        Y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, case in enumerate(list_cases_tmp):
            # Store sample
            x, y = dt.to_one_hot(case, self.n_classes)
            X[i,] = x
            Y[i,] = y

        return X, Y


class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors

    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        #x = Reshape((self.n_factors,))(x)
        return x


def RecommenderNet(n_in_classes, n_out_classes, embedding_size, n_layers, layer_size, dropout_rate, learning_rate):
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
    input_layer = Input(shape=(n_in_classes+1))
    #embeddings = Embedding(n_in_classes+1, embedding_size, embeddings_initializer='he_normal',
    #                      embeddings_regularizer=l2(1e-6))(input_layer)
    #embeddings_flat = Flatten()(embeddings)
    #embeddings_d = Dropout(dropout_rate)(embeddings_flat)

    #x = embeddings_d
    x = input_layer
    for i in range(0, n_layers):
        x = Dense(layer_size, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        layer_size = layer_size / 2

    x = Dense(n_out_classes+1, kernel_initializer='he_normal')(x)
    y = Activation('softmax')(x)
    model = Model(inputs=input_layer, outputs=y)
    opt = Adam(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    return model


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
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    train_data_path = args.train_file
    test_data_path = args.test_file
    is_five_fold_cross_validation = args.do_five_fold

    #train_data, val_data, classes = dt.read_data(train_data_path, 0.9)
    train_data, val_data, classes = dt.read_train_and_val_data(train_data_path, test_data_path)

    training_generator = DataGenerator(data=train_data, classes=classes, batch_size=2048, shuffle=True)
    validation_generator = DataGenerator(data=train_data, classes=classes, batch_size=128, shuffle=True)
    # initial recommender system
    recommender = RecommenderNet(
        n_in_classes=len(classes),
        n_out_classes=len(classes),
        embedding_size=4,
        n_layers=3,
        layer_size=4096,
        dropout_rate=0.05,
        learning_rate=0.001)
    # set params
    recommender.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              epochs=600,
                              use_multiprocessing=True,
                              workers=6,
                              validation_freq=10)

