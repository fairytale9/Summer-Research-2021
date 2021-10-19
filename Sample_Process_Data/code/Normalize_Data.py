#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np


def regularizeData(data):
    mean = data[:, 0:-1].mean(axis=0)
    data[:, 0:-1] -= mean
    std = data[:, 0:-1].std(axis=0)
    data[:, 0:-1] /= std
    return data


path = '../codedData'
dirs = os.listdir(path)
for x in dirs:
    if os.path.splitext(x)[1] == ".npy":
        filePath = path + '/' + x     
        data_key = os.path.splitext(x)[0]
        data_value = np.load(filePath) 
        data_after_regularization = regularizeData(data_value)
        np.save('../normalizedData/'+data_key+'.npy', data_after_regularization)


