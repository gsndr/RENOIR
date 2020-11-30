# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:50:57 2019

@author: miche
"""
import numpy as np


def reshapeFeature(x):
    feature = x.reshape(1, -1)

    return feature



def getXY(df):
    target = ['classification']
    Y = df[target]

    X = df.drop(target, axis=1)
    X = X.values

    return X, Y

def getResult(t,cm, N_CLASSES):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / N_CLASSES
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [t,tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR]
    return r
    

