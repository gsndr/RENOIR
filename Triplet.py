import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

np.random.seed(12)
import tensorflow as tf


tf.random.set_seed(12)
import keras.layers as kl


from keras import callbacks

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from keras.models import load_model
import time
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split

import TripletParams as triplet2


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

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def triplet_loss(self,y_true, y_pred):
        margin = K.constant(1)
        return K.mean(K.maximum(K.constant(0),
                    K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))
        #return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - 0.5 * (
        #            K.square(y_pred[:, 1, 0]) + K.square(y_pred[:, 2, 0])) + margin))

    def accuracy(self,y_true, y_pred):
        return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])



    def eucl_distance(self, vects):
        x, y, z = vects
        p_dist = K.sum(K.square(x - y), axis=1, keepdims=True)
        n_dist = K.sum(K.square(x - z), axis=1, keepdims=True)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def euclidean_distance(self,vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def create_base_network(self, input_shape):
        '''Base network to be shared (eq. to feature extraction).
        '''
        input = Input(shape=input_shape)
        x = Dense(128, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)




    def run(self):

        dsConf = self.ds
        print(dsConf.get('testPath') + ' dataset ')
        pathModels = dsConf.get('pathModels')
        pathPlot = dsConf.get('pathPlot')
        pathDataset = dsConf.get('pathDataset')
        configuration = self.config
        pathDatasetNumeric = dsConf.get('pathDatasetNumeric')
        pathDatasetEncoded = dsConf.get('pathDatasetEncoded')
        testPath = dsConf.get('testPath')
        n_classes = int(configuration.get('N_CLASSES'))

        VALIDATION_SPLIT = float(configuration.get('VALIDATION_SPLIT'))
        N_CLASSES = int(configuration.get('N_CLASSES'))
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

        LOAD_AUTOENCODER = int(configuration.get('LOAD_AUTOENCODER'))
        LOAD_AUTOENCODER1 = int(configuration.get('LOAD_AUTOENCODER1'))

        if (LOAD_AUTOENCODER == 0):
            autoencoder, best_time, encoder = ah.hypersearch(train_XN, train_YN, test_X, test_Y,
                                                             pathModels + 'autoencoderN.h5')

            self.file.write("Time Training Autoencoder: %s" % best_time)


        else:

            print('Load autoencoder')
            autoencoder = load_model(pathModels + 'autoencoderN.h5')
            autoencoder.summary()



        if (LOAD_AUTOENCODER1 == 0):
            autoencoderA, best_time, encoder = ah.hypersearch(train_XA, train_YA, test_X, test_Y,
                                                             pathModels + 'autoencoderA.h5')

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

        load_cnn = int(configuration.get('LOAD_CNN'))
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
        '''
        
        rows = [train_X, train_R, train_RA]
        rows = [list(i) for i in zip(*rows)]
        x_train = np.array(rows)
        '''
        print(x_train.shape)
        rows = [test_X, test_R, test_RA]
        rows = [list(i) for i in zip(*rows)]
        x_test = np.array(rows)
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler()
        '''
        scalers = {}
        for i in range(x_train.shape[1]):
             scalers[i] = preprocessing.MinMaxScaler()
             x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :]) 

        for i in range(x_test.shape[1]):
             x_test[:, i, :] = scalers[i].transform(x_test[:, i, :]) 
             
        print(x_train[0])
        '''
        
           
           
        

        if (load_cnn == 0):
            margin = 0.2
            callbacks_list = [
                callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, restore_best_weights=True),
            ]
            tic = time.time()

            print(train_Y.shape)
            # network definition
            #model, best_time = triplet.hypersearch(x_train, train_Y, x_test, test_Y,
             #                                      pathModels + 'siamese.h5')
            '''
            base_network = self.create_base_network(input_shape)

            input_a = Input(shape=input_shape)
            input_b = Input(shape=input_shape)
            input_c= Input(shape=input_shape)


            processed_a = base_network(input_a)
            processed_b = base_network(input_b)
            processed_c = base_network(input_c)

            # The Lamda layer produces output using given function. Here its Euclidean distance.

            positive_dist = kl.Lambda(self.euclidean_distance, name='pos_dist')([processed_a, processed_b])
            negative_dist = kl.Lambda(self.euclidean_distance, name='neg_dist')([processed_a, processed_c])
            #tertiary_dist = kl.Lambda(self.euclidean_distance, name='ter_dist')([processed_b, processed_c])

            # This lambda layer simply stacks outputs so both distances are available to the objective
            distance = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')(
                [positive_dist, negative_dist])

          
            # Add a dense layer with a sigmoid unit to generate the similarity score
            #prediction = Dense(1, activation='sigmoid')(distance)

            model = Model([input_a, input_b,input_c], distance)
            plot_model(model, show_shapes=True, show_layer_names=True, to_file='02 model.png')

            # train
            rms =Adam()
            # Variable Learning Rate per Layers
            lr_mult_dict = {}
            last_layer = ''
            for layer in model.layers:
                # comment this out to refine earlier layers
                # layer.trainable = False
                # print layer.name
                lr_mult_dict[layer.name] = 1
                # last_layer = layer.name
            lr_mult_dict['t_emb_1'] = 100
            base_lr = 0.0001
            momentum = 0.9
            rms = Adam(0.0006)
            model.compile(loss=self.triplet_loss, optimizer=rms, metrics=[self.accuracy])
            model.summary()

            XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                  test_size=0.2)  # before model building

            print(XTraining[:,0].shape)
            model.fit([XTraining[:, 0], XTraining[:, 1], XTraining[:, 2]], YTraining,  batch_size=512,
                       epochs=150,
                       callbacks=callbacks_list,
                       validation_data=([XValidation[:, 0], XValidation[:, 1], XValidation[:, 2]], YValidation))
            '''


            #model.save(pathModels + 'siameseTriplet.h5')
            toc = time.time()
            self.file.write("Time Fitting CNN : " + str(toc - tic))
            self.file.write('\n')

            #modelSoftmax, best_time2 = triplet2.hypersearch(x_train, train_Y, x_test, test_Y,
            #                                                pathModels + 'siamese.h5')

            #modelSoftmax.save(pathModels + 'siameseSoftmax.h5')

            model, best_time2 = triplet2.hypersearch(x_train, train_Y, x_test, test_Y,
                                                            pathModels + 'softmax.h5')

            model.save(pathModels + 'maxPlus.h5')

        else:
            print('Load CNN')
            modelName = 'softplus.h5'
            print(pathModels)
            model = load_model(pathModels + modelName, compile=False)
            model.summary()

        tic_prediction_classifier = time.time()
        
       
        y_predT = model.predict([train_X, train_R, train_RA])
        pred= y_predT[:, 0, 0] < y_predT[:, 1, 0]
        y_predT = np.squeeze(y_predT, axis = 2)  
        print(y_predT.shape)
        cm = confusion_matrix(train_Y, pred)
        print(cm)
        
        
        
        predDf=pd.DataFrame(data = y_predT
             , columns = ['d_N', 'dA'])
        
        resultPCA = pd.concat([predDf,train_Y], axis=1, sort=False)
        resultPCA['pred']=pred
        resultPCA.to_csv(testPath + '_distance.csv', index=False)
        exit()
        

        print('Softmax on test set')
        # create pandas for results
        columns = ['Algorithm','TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        results = pd.DataFrame(columns=columns)
        y_pred = model.predict([test_X, test_R, test_RA])
        print(y_pred)
        pred= y_pred[:, 0, 0] < y_pred[:, 1, 0]
        print(pred)
        
        
        
        y_pred = np.squeeze(y_pred, axis = 2)  
        print(y_pred)
        predDf=pd.DataFrame(data = y_pred
             , columns = ['d_N', 'dA'])
        
        
        resultPCA = pd.concat([predDf,test_Y], axis=1, sort=False)
        resultPCA['pred']=pred
        resultPCA.to_csv(testPath + '_distanceTest.csv', index=False)
        #exit()
        
        #exit()
        cm = confusion_matrix(test_Y, pred)
        r = getResult('Embedding',cm, n_classes)
        dfResults = pd.DataFrame([r], columns=columns)
       

        print(dfResults)
        exit()
        results.append(dfResults)
        print(results)

        toc_prediction_classifier = time.time()
        time_prediction_classifier = (toc_prediction_classifier - tic_prediction_classifier)
        self.file.write("Time for predictions: %s " % (time_prediction_classifier))
        print(y_pred)
        '''
        threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for t in threshold:
            pred = y_pred.ravel() < t

            cm = confusion_matrix(test_Y, pred)
            r = getResult(cm, n_classes)

            dfResults = pd.DataFrame([r], columns=columns)

            print(dfResults)

            results = results.append(dfResults, ignore_index=True)

        
        

        #Softmax

        y_pred = modelSoftmax.predict([test_X, test_R, test_RA])
        pred = y_pred[:, 0, 0] < y_pred[:, 1, 0]
        cm = confusion_matrix(test_Y, pred)
        r = getResult('softmax', cm, n_classes)
        dfResults = pd.DataFrame([r], columns=columns)
        results.append(dfResults)
        print(results)

        results = pd.DataFrame(columns=columns)
        y_pred = modelSigmoid.predict([test_X, test_R, test_RA])
        pred = y_pred[:, 0, 0] < y_pred[:, 1, 0]
        cm = confusion_matrix(test_Y, pred)
        r = getResult('sigmoid', cm, n_classes)
        dfResults = pd.DataFrame([r], columns=columns)
        results.append(dfResults)
        print(results)
        '''

        results.to_csv(testPath + '_resultsTriplet.csv', index=False)
        self.file.close()


















