#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from Training_Val_Test_DataLoader import tDataLoader


###load model
print('\n ---> Loading model ... \n')
model = load_model('../model/lstm/weights.17-1.08.hdf5')


print("\n ---> Testing ... \n")
nNegative, nPositive = 0, 0
cNegative, cPositive = 0.0, 0.0

for data, label in tDataLoader.loadTestData(batch_size=1):
	predList = model.predict_on_batch(data)
	y_pred = np.argmax(predList)
	y_true = np.argmax(label)

	if y_true == 0:
		nNegative += 1
		if y_pred == 0:
			cNegative += 1
	elif y_true == 1:
		nPositive += 1
		if y_pred == 1:
			cPositive += 1


tNegativeRate = cNegative / nNegative
tPositiveRate = cPositive / nPositive
accNorm = (tNegativeRate + tPositiveRate) / 2.0

print(nNegative, cNegative)
print(nPositive, cPositive)

print("AccNorm = ", accNorm)
print("tNegativeRate = ", tNegativeRate)
print("tPositiveRate = ", tPositiveRate)
