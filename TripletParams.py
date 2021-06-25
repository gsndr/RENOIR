from __future__ import print_function
import numpy as np

# random seeds must be set before importing keras & tensorflow
my_seed = 12
np.random.seed(my_seed)
import random
import keras.layers as kl

random.seed(my_seed)


from hyperopt import Trials, STATUS_OK, tpe

from keras.models import Model


from keras import callbacks
from hyperas import optim
from hyperas.distributions import choice, uniform


import keras.backend as K



from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam
from tabulate import tabulate
import global_config
from sklearn.model_selection import train_test_split
import time
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def getResult(cm):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = (tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR)
    return r


def euclidean_distance(vects):
    x, y = vects
    print(x)
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def triplet_loss(y_true, y_pred):
    margin = K.constant(0)
    loss=y_pred[:, 0:1] - y_pred[:, 1:2]
    return K.log(1 + K.exp(loss))
    

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:, 0:1] < y_pred[:, 1:2])

def create_base_network(input_shape, dense_filter1, dense_filter2, dense_filter3, dropout1, dropout2):
    input = Input(shape=input_shape)
    x = Dense(dense_filter1, activation='relu')(input)
    x = Dropout(dropout1)(x)
    x = Dense(dense_filter2, activation='relu')(x)
    x = Dropout(dropout2)(x)
    x = Dense(dense_filter3, activation='relu')(x)
    x = Dense(512, activation='sigmoid')(x)
    return Model(input, x)



def data():
    y_train = global_config.train_Y
    y_test = global_config.test_Y
    x_train = global_config.train_X
    x_test = global_config.test_X
    global_config.savedScore = []
    global_config.savedTrain = []
    print(x_train.shape)
    return x_train, y_train, x_test, y_test


def getBatchSize(p, bs):
    return bs[p]


def Siamese(x_train, y_train, x_test, y_test):
    t =x_train[:,0]
    input_shape = t.shape[1:]
    print(input_shape)

    # network definition
    dropout1 = {{uniform(0, 1)}}
    dropout2 = {{uniform(0, 1)}}
    dense_filter1 = {{choice([128, 256,512])}}
    dense_filter2 = {{choice([128, 256,512])}}
    dense_filter3 = {{choice([128, 256,512])}}
    base_network = create_base_network(input_shape, dense_filter1, dense_filter2, dense_filter3, dropout1, dropout2)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    input_c = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    processed_c = base_network(input_c)

    # The Lamda layer produces output using given function. Here its Euclidean distance.

    positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([processed_a, processed_b])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([processed_a, processed_c])

    # This lambda layer simply stacks outputs so both distances are available to the objective
    distance = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')(
        [positive_dist, negative_dist])
    #distance = Lambda(lambda x: K.l2_normalize(distance, axis=1))(distance)

    model = Model([input_a, input_b,input_c], distance)

    # train
    adam = Adam(lr={{uniform(0.0001, 0.01)}})

    model.compile(loss=triplet_loss, optimizer=adam, metrics=[accuracy])
    model.summary()
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20,
                                restore_best_weights=True),
    ]

    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building

    tic = time.time()

    h=model.fit([XTraining[:, 0], XTraining[:, 1], XTraining[:, 2]], YTraining, batch_size={{choice([32, 64, 128, 256, 512])}},
              epochs=150,
              callbacks=callbacks_list,
               verbose=2,
              validation_data=([XValidation[:, 0], XValidation[:, 1], XValidation[:, 2]], YValidation))

    toc = time.time()

    scores = [h.history['val_accuracy'][epoch] for epoch in range(len(h.history['loss']))]
    score = max(scores)
    print('Score', score)
    y_pred = model.predict([x_test[:, 0], x_test[:, 1],x_test[:, 2]])
    pred= y_pred[:, 0:1].ravel() < y_pred[:, 1:2].ravel()

    cmTest = confusion_matrix(y_test, pred)
    r=getResult(cmTest)
    print(r)
    f1_test =r[8]
    print(f1_test)
    global_config.savedScore.append(cmTest)
    Y_pred = model.predict([XValidation[:, 0], XValidation[:, 1],XValidation[:, 2]])
    pred = Y_pred[:, 0:1].ravel() < Y_pred[:, 1:2].ravel()
    cmTrain = confusion_matrix(YValidation, pred)
    r = getResult(cmTrain)
    print(r)
    f1_train = r[8]

    global_config.savedTrain.append(cmTrain)
    print('Best score', global_config.best_score2)

    if global_config.best_score2 < score:
        global_config.best_score2 = score
        global_config.best_model = model
        global_config.best_numparameters = model.count_params()
        global_config.best_time = toc - tic
    print(len(h.history['loss']))
    return {'loss': -score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']),
            'n_params': model.count_params(), 'model': global_config.best_model, 'time': toc - tic}


def hypersearch(train_X1, train_Y1, test_X1, test_Y1, modelName):
    trials = Trials()
    global_config.train_X = train_X1
    global_config.train_Y = train_Y1
    global_config.test_X = test_X1
    global_config.test_Y = test_Y1
    global_config.savedScore = []
    global_config.savedTrain = []
    global_config.best_score = 0
    global_config.best_model = None
    global_config.best_time = 0

    bs = [32, 64, 128, 256, 512]
    nr = [128, 256,512]
    best_run, best_model = optim.minimize(model=Siamese,
                                          data=data,
                                          functions=[create_base_network, euclidean_distance,
                                                     triplet_loss, eucl_dist_output_shape,  accuracy, getResult],
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=trials)
    outfile = open(modelName+'softplus.csv', 'w')
    outfile.write("\nHyperopt trials")


    outfile.write(
        "\ntid , val_acc , learning_rate , Dropout1 , Droput2, filter1,filter2, filter3, batch_size, time, epochs, TP_VAL, FN_VAL, FP_VAL, TN_VAL, OA_VAL, P_VAL, R_VAL, F1_VAL, TP_TEST, FN_TEST, FP_TEST,     TN_TEST, OA_TEST,P_TEST, R_TEST, F1_TEST ")
    for trial, test, train in zip(trials.trials, global_config.savedScore, global_config.savedTrain):
        t = getResult(test)
        v = getResult(train)

        outfile.write(
            "\n%s , %f , %f , %f, %f , %s, %s, %s, %s ,%s, %d, %d , %d , %d, %d, %f , %f , %f, %f ,%d , %d , %d, %d, %f, %f, %f, %f  " % (
            trial['tid'],
            abs(trial['result']['loss']),
            trial['misc']['vals']['lr'][0],
            trial['misc']['vals']['dropout1'][0],
            trial['misc']['vals']['dropout1_1'][0],
            getBatchSize(trial['misc']['vals']['dense_filter1'][0], nr),
            getBatchSize(trial['misc']['vals']['dense_filter1_1'][0], nr),
            getBatchSize(trial['misc']['vals']['dense_filter1_2'][0], nr),
            getBatchSize(trial['misc']['vals']['batch_size'][0], bs),
            trial['result']['time'],
            trial['result']['n_epochs'],
            v[0], v[1], v[2], v[3], v[4], v[6], v[7], v[8],
            t[0], t[1], t[2], t[3], t[4], t[6], t[7], t[8]
            ))

    outfile.write('\nBest model:\n ')
    outfile.write(str(best_run))

    global_config.best_model.save(modelName)
    return global_config.best_model, global_config.best_time
