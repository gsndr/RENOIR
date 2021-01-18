import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

np.random.seed(12)
import tensorflow as tf


tf.random.set_seed(12)



from keras import callbacks

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from keras.models import load_model
import time
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda


import TripletParams as triplet


np.set_printoptions(suppress=True)

from Utils import  getResult

import AutoencoderParams as ah

from DatasetConfig import Datasets

import Preprocessing as prp
from keras.engine.topology import Layer
from keras.utils import plot_model


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
class Execution():
    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig
        fileOutput = self.ds.get('pathTime') + 'result' + self.ds.get('testPath') + '.txt'
        self.file = open(fileOutput, 'w')
        self.file.write('Result time \n')
        self.file.write('\n')


    def run(self):

        dsConf = self.ds
        print(dsConf.get('testPath') + ' dataset ')
        pathModels = dsConf.get('pathModels')
        pathPlot = dsConf.get('pathPlot')
        pathDataset = dsConf.get('pathDataset')
        configuration = self.config
        pathDatasetNumeric = dsConf.get('pathDatasetNumeric')
        testPath = dsConf.get('testPath')
        n_classes = int(configuration.get('N_CLASSES'))


        pd.set_option('display.expand_frame_repr', False)

        # Preprocessing phase from original to numerical dataset
        PREPROCESSING1 = int(configuration.get('PREPROCESSING1'))

        ds = Datasets(dsConf)
        if (PREPROCESSING1 == 1):
            tic_preprocessing1 = time.time()

            prp.toNumeric(ds)
            toc_preprocessing1 = time.time()
            time_preprocessing1 = toc_preprocessing1 - tic_preprocessing1
            self.file.write("Time Preprocessing: %s" % (time_preprocessing1))

        train = pd.read_csv(pathDatasetNumeric + 'Train_standard.csv')
        test = pd.read_csv(pathDatasetNumeric + 'Test.csv')

        self._clsTrain = ds.getClassification(train)
        print(self._clsTrain)
        train_X, train_Y = prp.getXY(train, self._clsTrain)
        test_X, test_Y = prp.getXY(test, self._clsTrain)

        train_normal = train[(train[self._clsTrain] == 1)]
        print("train normal:", train_normal.shape)
        train_anormal = train[(train[self._clsTrain] == 0)]
        print("train anormal:", train_anormal.shape)
        test_normal = test[(test[self._clsTrain] == 1)]

        train_XN, train_YN = prp.getXY(train_normal, self._clsTrain)
        train_XA, train_YA = prp.getXY(train_anormal, self._clsTrain)

        LOAD_AUTOENCODER = int(configuration.get('LOAD_AUTOENCODER_N'))
        LOAD_AUTOENCODER1 = int(configuration.get('LOAD_AUTOENCODER_A'))

        if (LOAD_AUTOENCODER == 0):
            autoencoder, best_time, encoder = ah.hypersearch(train_XN, train_YN, test_X, test_Y,
                                                             pathModels + 'autoencoderN')

            self.file.write("Time Training Autoencoder: %s" % best_time)


        else:

            print('Load autoencoder')
            autoencoder = load_model(pathModels + 'autoencoderN.h5')
            autoencoder.summary()



        if (LOAD_AUTOENCODER1 == 0):
            autoencoderA, best_time, encoder = ah.hypersearch(train_XA, train_YA, test_X, test_Y,
                                                             pathModels + 'autoencoderA')

            self.file.write("Time Training Autoencoder Attacks: %s" % best_time)


        else:

            print('Load autoencoder')
            autoencoderA = load_model(pathModels + 'autoencoderA.h5')
            autoencoderA.summary()

        ''' Encoded dataset creation'''
        tic_preprocessingAutoencoder = time.time()
        train_R = autoencoder.predict(train_X)
        test_R = autoencoder.predict(test_X)
        input_shape = train_X.shape[1:]
        print(input_shape)
        train_RA =autoencoderA.predict(train_X)
        test_RA = autoencoderA.predict(test_X)
        toc_preprocessingAutoencoder = time.time()
        self.file.write(
            "Time Creations Autoencoder Dataset: %s" % (toc_preprocessingAutoencoder - tic_preprocessingAutoencoder))

        load_cnn = int(configuration.get('LOAD_NN'))
        input_shape = train_X.shape[1:]
        train_RN =autoencoder.predict(train_XN)
        train_RAN = autoencoderA.predict(train_XN)
        train_RNA = autoencoder.predict(train_XA)
        train_RAA = autoencoderA.predict(train_XA)
        '''
        x_train1 = np.append(train_XN, train_XA, axis=0)
        x_train2=np.append(train_RN, train_RNA)
        x_train3=np.append(train_RAN,train_RAA)
        rows=[x_train1, x_train2,x_train3]
        print(rows)
        '''
        rowsA = [train_XA, train_RAA, train_RNA]
        rowsA = [list(i) for i in zip(*rowsA)]
        rowsN= [train_XN, train_RN, train_RAN]
        rowsN = [list(i) for i in zip(*rowsN)]
        
        
        
        y_train = np.append(train_YN, train_YA, axis=0)
        print(y_train.shape)
        train_XR =np.append(np.array(rowsN), np.array(rowsA), axis=0)
        
        

        #X_trainA=np.array(rowsA)

        x_train = train_XR.reshape(train_XR.shape[0], train_XR.shape[1], train_XR.shape[2])
        print(x_train.shape)

        print(x_train[0])

        

        print(x_train.shape)
        rows = [test_X, test_R, test_RA]
        rows = [list(i) for i in zip(*rows)]
        x_test = np.array(rows)

        
           
           
        

        if (load_cnn == 0):
            margin = 0.2
            callbacks_list = [
                callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, restore_best_weights=True),
            ]
            tic = time.time()

            print(train_Y.shape)

            toc = time.time()
            self.file.write("Time Fitting CNN : " + str(toc - tic))
            self.file.write('\n')


            model, best_time2 = triplet.hypersearch(x_train, y_train, test_X, test_Y,
                                                            pathModels)

            model.save(pathModels + 'softplus.h5')

        else:
            print('Load CNN')
            modelName = 'maxPlus.h5'
            print(pathModels)
            model = load_model(pathModels + modelName, compile=False)
            model.summary()

        tic_prediction_classifier = time.time()

        

        print('Softmax on test set')
        # create pandas for results
        columns = ['Algorithm','TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        results = pd.DataFrame(columns=columns)
        y_pred = model.predict([test_X, test_R, test_RA])
        print(y_pred)
        pred= y_pred[:, 0, 0] < y_pred[:, 1, 0]

        
        
        



        cm = confusion_matrix(test_Y, pred)
        r = getResult('Embedding',cm, n_classes)
        results = pd.DataFrame([r], columns=columns)
        print(results)
       



        toc_prediction_classifier = time.time()
        time_prediction_classifier = (toc_prediction_classifier - tic_prediction_classifier)
        self.file.write("Time for predictions: %s " % (time_prediction_classifier))



        results.to_csv(testPath + '_resultsTriplet.csv', index=False)
        self.file.close()


















