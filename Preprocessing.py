import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale, MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder





#one hot encoder
def ohe(train, test, categories):
        all_data = pd.concat((train, test))
        for column in all_data.select_dtypes(include=[np.object]).columns:
            train[column] = train[column].astype(
                pd.api.types.CategoricalDtype(categories=all_data[column].unique()))
            test[column] = test[column].astype(pd.api.types.CategoricalDtype(categories=all_data[column].unique()))

        for cat in categories:
            trainDum = pd.get_dummies(train[cat], prefix=cat)
            testDum = pd.get_dummies(test[cat], prefix=cat)
            train = pd.concat([train, trainDum.reindex(sorted(trainDum.columns), axis=1)], axis=1)
            test = pd.concat([test, testDum.reindex(sorted(testDum.columns), axis=1)], axis=1)
            train = train.drop(cat, axis=1)
            test = test.drop(cat, axis=1)

        return train, test

#Scaler
def scaler(train, test, listContent):
    listContent = list(listContent)
    scaler = MinMaxScaler()  # for NSL-KDD
    frames = [train[listContent], test[listContent]]
    scaler.fit(train[listContent].values)  # Remember to only fit scaler to training data
    train[listContent] = scaler.transform(train[listContent])
    test[listContent] = scaler.transform(test[listContent])
    return train, test

def scalerCICIDS(train, testList, listContent):
    tests = list()
    listContent = list(listContent)
    scaler = MinMaxScaler()
    scaler.fit(train[listContent].values)  # Remember to only fit scaler to training data
    train[listContent] = scaler.transform(train[listContent])
    for test in testList:
        test[listContent] = scaler.transform(test[listContent])
        tests.append(test)
    return train, tests


def toNumeric(ds):
    ds.preprocessing()



def getXY(train, clsTrain):
    clssList = train.columns.values
    #target = [i for i in clssList if i.startswith(' classification')]
    target=[i for i in clssList if i.startswith(clsTrain)]
    # remove label from dataset to create Y ds
    train_Y=train[target]
    train_X = train.drop(target, axis=1)
    train_X=train_X.values
    return train_X, train_Y

def getXYCICIDS(train, tests, clsTrain):
    clssList = train.columns.values
    # target = [i for i in clssList if i.startswith(' classification')]
    target = clsTrain

    # remove label from dataset to create Y ds
    train_Y = train[target]
    train_X = train.drop(target, axis=1)
    train_X = train_X.values
    test_Y = list()
    test_X = list()
    for t in tests:
        print("uno")
        print(t.head(8))
        t_Y = t[target]
        t_X = t.drop(target, axis=1)
        t_X = t_X.values
        test_Y.append(t_Y)
        test_X.append(t_X)

    return train_X, train_Y, test_X, test_Y




