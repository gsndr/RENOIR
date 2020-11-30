from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe, hp
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, concatenate, LSTM, Conv2D, Conv1D, \
    MaxPooling1D, MaxPooling2D, ZeroPadding2D, Activation, Add, AveragePooling2D
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.regularizers import l1
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform
import pandas as pd
from keras import regularizers

np.random.seed(12)
import tensorflow as tf
tf.random.set_seed(12)


from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam
from keras import callbacks
import time

import global_config
from sklearn.model_selection import train_test_split





def data():





    y_train = global_config.train_Y
    y_test = global_config.test_Y
    x_train = global_config.train_X
    x_test = global_config.test_X

    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def getBatchSize(p, bs):
    return bs[p]


def Autoencoder(x_train, y_train, x_test, y_test):
    input_shape = (x_train.shape[1],)
    input2 = Input(input_shape)


    encoded = Dense(32, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod1')(input2)
    encoded = Dense(16, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod2')(encoded)

    encoded= Dropout({{uniform(0, 1)}})(encoded)
    decoded = Dense(32, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder1')(encoded)
    decoded = Dense(x_train.shape[1], activation='linear',
                    kernel_initializer='glorot_uniform',
                    name='decoder3')(decoded)


    model = Model(inputs=input2, outputs=decoded)
    model.summary()

    adam=Adam(lr={{uniform(0.0001, 0.01)}})
    model.compile(loss='mse', metrics=['acc'],
                  optimizer=adam)
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]
    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building
    print(XTraining.shape)
    tic = time.time()
    history= model.fit(XTraining, XTraining,
                      batch_size={{choice([32,64, 128,256,512])}},
                      epochs=150,
                      verbose=2,
                      callbacks=callbacks_list,
                      validation_data=(XValidation,XValidation))

    toc = time.time()


    score = np.amin(history.history['val_loss'])
    print('Best validation loss of epoch:', score)


    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    score = min(scores)
    print('Score',score)


    print('Best score',global_config.best_score)




    if global_config.best_score > score:
        global_config.best_score = score
        global_config.best_model = model
        global_config.best_numparameters = model.count_params()
        global_config.best_time = toc - tic



    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']), 'n_params': model.count_params(), 'model': global_config.best_model, 'time':toc - tic}

def SparseAutoencoder(x_train, y_train, x_test, y_test):
    input_shape = (x_train.shape[1],)
    input2 = Input(input_shape)


    encoded = Dense(32, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod1')(input2)
    encoded = Dense(16, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod2',activity_regularizer=regularizers.l1(10e-5))(encoded)

    encoded= Dropout({{uniform(0, 1)}})(encoded)
    decoded = Dense(32, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder1')(encoded)
    decoded = Dense(x_train.shape[1], activation='linear',
                    kernel_initializer='glorot_uniform',
                    name='decoder3')(decoded)


    model = Model(inputs=input2, outputs=decoded)
    model.summary()

    adam=Adam(lr={{uniform(0.0001, 0.01)}})
    model.compile(loss='mse', metrics=['acc'],
                  optimizer=adam)
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]
    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building
    print(XTraining.shape)
    tic = time.time()
    history= model.fit(XTraining, XTraining,
                      batch_size={{choice([32,64, 128,256,512])}},
                      epochs=150,
                      verbose=20,
                      callbacks=callbacks_list,
                      validation_data=(XValidation,XValidation))

    toc = time.time()


    score = np.amin(history.history['val_loss'])
    print('Best validation loss of epoch:', score)


    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    score = min(scores)
    print('Score',score)


    print('Best score',global_config.best_score)




    if global_config.best_score > score:
        global_config.best_score = score
        global_config.best_model = model
        global_config.best_numparameters = model.count_params()
        global_config.best_time = toc - tic



    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']), 'n_params': model.count_params(), 'model': global_config.best_model, 'time':toc - tic}



def hypersearch(train_X1, train_Y1, test_X1, test_Y1, pathModel):

    global_config.train_X =  train_X1
    global_config.train_Y =train_Y1
    global_config.test_X = test_X1
    global_config.test_Y =test_Y1



    trials = Trials()

    bs = [32, 64, 128, 256, 512]
    best_run, best_model = optim.minimize(model=Autoencoder,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=trials)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    outfile = open(pathModel+'.csv', 'w')
    outfile.write("\nHyperopt trials")

    outfile.write("\ntid , loss , learning_rate , Dropout , batch_size, time")
    for trial in trials.trials:
        # outfile.write(str(trial))
        outfile.write("\n%s , %f , %f , %s , %s, %s" % (trial['tid'],
                                                        trial['result']['loss'],
                                                        trial['misc']['vals']['lr'][0],
                                                        trial['misc']['vals']['Dropout'],
                                                        getBatchSize(trial['misc']['vals']['batch_size'][0], bs),
                                                        trial['result']['time']
                                                        ))



    outfile.write('\nBest model:\n ')
    outfile.write(str(best_run))
    outfile.flush()
    global_config.best_model.save(pathModel+'.h5')
    encoder = Model(inputs=global_config.best_model.input, outputs=global_config.best_model.get_layer('encod2').output)
    encoder.summary()

    return global_config.best_model, global_config.best_time, encoder
