#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from Training_Val_Test_DataLoader import train_gen, val_gen

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Multiply, LSTM, GRU
#from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint


modelDirectory = '../model/lstm'

###build model
'''
#conv1d
model = Sequential()
model.add(Conv1D(32, 7, input_shape=(60, 169)))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 7))
#model.add(Flatten())
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation='softmax'))
'''

#LSTM
input_tensor = Input(shape=(60, 169))
output = LSTM(32, return_sequences=True)(input_tensor)
output = LSTM(128)(output)
#output = Flatten()(output)
output = Dense(2, activation='softmax')(output)
model = Model(input_tensor, output)


model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics='acc')

checkpointer = ModelCheckpoint(filepath=os.path.join(modelDirectory, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), verbose=0)

history = model.fit(train_gen,
                    steps_per_epoch=35,
                    epochs=30, 
                    validation_data=val_gen,
                    validation_steps=36,
                    callbacks = checkpointer)

#model.save('../model/model_2.hdf5')


###plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()