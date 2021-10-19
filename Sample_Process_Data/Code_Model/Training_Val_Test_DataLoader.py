#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import random


###load numpy datas
data_list = []

path = '../processed data/normalized'
dirs = os.listdir(path)
for x in dirs:
    if os.path.splitext(x)[1] == ".npy":
        filePath = path + '/' + x        
        data_value = np.load(filePath)   
        data_list.append(data_value)



trainDataPool, trainLabelPool = [], []
valDataPool, valLabelPool = [], []
testDataPool, testLabelPool = [], []

train_positive_label = 0
val_positive_label = 0
test_positive_label = 0

#adjustable parameters
threshold, period, overlap = 0, 60, 30

random.seed(4)

def processData(data):
    global train_positive_label, val_positive_label, test_positive_label
    for idx in range(0, len(data), period-overlap):
        if idx+period > len(data):
            break
        DataValue = data[idx:idx+period, 0:-1]
        totalLabel = np.sum(data[idx:idx+period, -1])
        if totalLabel > threshold:
            DataLabel = np.array([0, 1])
        else:
            DataLabel = np.array([1, 0])
        
        rn = random.random()
        if rn < 0.6:
            trainDataPool.append(DataValue)
            trainLabelPool.append(DataLabel)
            if totalLabel > threshold:
                train_positive_label += 1
        elif rn < 0.8:
            valDataPool.append(DataValue)
            valLabelPool.append(DataLabel)
            if totalLabel > threshold:
                val_positive_label += 1
        else:
            testDataPool.append(DataValue)
            testLabelPool.append(DataLabel)
            if totalLabel > threshold:
                test_positive_label += 1
            

                    
for data in data_list:
    processData(data)
    
    
trainDataPool = np.array(trainDataPool)
trainLabelPool = np.array(trainLabelPool)
valDataPool = np.array(valDataPool)
valLabelPool = np.array(valLabelPool)
testDataPool = np.array(testDataPool)
testLabelPool = np.array(testLabelPool)


def generator(data, label, batch_size=20, shuffle=False):
    max_index = len(data)
    min_index = 0
    while 1:
        if shuffle:
            rows = np.random.randint(0, max_index, size=batch_size)
        else:
            if min_index+batch_size > max_index:
                min_index = np.random.randint(0, batch_size)
            rows = np.arange(min_index, min_index+batch_size)
            min_index += batch_size
        values = data[rows]
        labels = label[rows]
        yield values, labels
        
train_gen = generator(trainDataPool, trainLabelPool, batch_size=30)
val_gen = generator(valDataPool, valLabelPool, batch_size=10)


class testDataLoader(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def loadTestData(self, batch_size=1):
        data = self.data
        label = self.label
        data_batch, label_batch = [], []
        nInBatch = 0
        for idx in range(len(data)):
            data_batch.append(data[idx])
            label_batch.append(label[idx])
    
            nInBatch += 1
            if nInBatch >= batch_size:
                yield np.array(data_batch), np.array(label_batch)
                data_batch, label_batch = [], []
                nInBatch = 0

        if nInBatch >= batch_size:
                yield np.array(data_batch), np.array(label_batch)


tDataLoader = testDataLoader(testDataPool, testLabelPool)

'''
###check the distribution of data
total_len = 0
for data in data_list:
    total_len += (len(data) - 60) // 30 + 1 
print(total_len)

print('train:', len(trainDataPool), len(trainLabelPool), train_positive_label)
print('val:', len(valDataPool), len(valLabelPool), val_positive_label)
print('test:', len(testDataPool), len(testLabelPool), test_positive_label)
'''