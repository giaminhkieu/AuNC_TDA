from ElementsInfo import *
import os
import numpy as np
from PersistentImageCode import *
from VariancePersistCode import *
from GenerateImagePI import *

import pandas as pd

DatasetAuNC = pd.read_pickle('DatasetAuNC100.pkl')
DatasetTM = pd.read_pickle('DatasetTM100.pkl')

"""
Resnet codes
"""
import math
from tensorflow import keras
from tensorflow.keras import layers

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)


'''
Initializing and data appending for 3-stream Model
'''
import random
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
x1, x2, x3, y = [], [], [], []

for i in range(len(DatasetAuNC)):
    x1.append(np.asarray([DatasetAuNC["core"][i],
                          DatasetAuNC["Tetrahedral_count"][i],
                          DatasetAuNC["Unconnected_triangles_count"][i],
                          DatasetAuNC["Triangles_with_1_shared_vertex_count"][i],
                          DatasetAuNC["Triangles_with_2_shared_vertices_count"][i]]))
    
    x2.append(np.asarray(DatasetAuNC["PersImg"][i]).reshape(100, 100, 1))
    x3.append([DatasetAuNC["Charge"][i]])
    y.append(float(DatasetAuNC["gap"][i]))

y = np.array(y)
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

'''
Optimizing hyperparams
'''

from sklearn.model_selection import ParameterGrid

learning_rates = [0.001, 0.01, 0.1]
epochs = [5, 15, 30]
param_grid = {'learning_rate': learning_rates, 'epochs': epochs}

for params in ParameterGrid(param_grid):
    print(params)

    # Train-validation split
    x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=42)

    # Clear the session and define the model
    tf.keras.backend.clear_session()

    input2 = Input(shape=(100,100,1))
    conv_1 = Conv2D(16, (3, 3), activation='relu')(input2)
    maxpool_1 = MaxPooling2D((2, 2))(conv_1)
    conv_2 = Conv2D(16, (3, 3), activation='relu')(maxpool_1)
    maxpool_2 = MaxPooling2D((2, 2))(conv_2)
    flatten = Flatten()(maxpool_2)
    dense_2 = Dense(32, activation='relu', name="dense_a")(flatten)
    dense_3 = Dense(16, activation='relu', name="dense_b")(dense_2)
    dense_4 = Dense(1, name="dense_c")(dense_3)

    model = Model(inputs=input2, outputs=dense_4)
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='mean_absolute_error',
                  metrics=['mean_squared_error'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    model_checkpoint = ModelCheckpoint(filepath='best_model_TM.h5',
                                       save_best_only=True,
                                       save_weights_only=False,
                                       monitor='val_loss',
                                       mode='min',
                                       verbose=1)

    # Train the model and save the best one based on validation loss
    model.fit(x2_train, y_train,
              batch_size=128,
              epochs=params['epochs'],
              verbose=1,
              validation_data=(x2_test, y_test),
              callbacks=[early_stopping, model_checkpoint])

    # Load the best model
    model = load_model('best_model_TM.h5')

    # Evaluate the model on the test set
    predicted_arr = []
    true_arr = []
    TotalError = 0
    MSE = 0

    for i in range(len(x2_test)):
        y_pred = model.predict(np.asarray([x2_test[i]]))
        predicted_arr.append(y_pred)
        true_arr.append(float(y_test[i]))
        abs_error = abs(float(y_pred)- float(y_test[i]))
        TotalError += abs_error
        MSE += float(abs_error**2)

    size = len(y_test)
    print(80*"-")
    print('model structure: (16,3,3), (2,2), (16,3,3), (2,2), flatten, 32, 16')
    print(f"Mean Absolute Error = {TotalError/size}")
    print("RMSE =", np.sqrt(MSE/size))
    print(80*"-")

