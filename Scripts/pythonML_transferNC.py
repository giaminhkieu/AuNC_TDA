from ElementsInfo import *
import os
import numpy as np
from PersistentImageCode import *
from VariancePersistCode import *
from GenerateImagePI import *

import pandas as pd

DatasetAuNC = pd.read_pickle('DatasetAuNC_290723.pkl')
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

target_variable = "mu"
model_name = "010823_transfer3stream_" + target_variable
model_load_name = "010823_transferTM_" + target_variable + ".h5"


for i in range(len(DatasetAuNC)):
    x1.append(np.asarray([DatasetAuNC["core"][i],
                          DatasetAuNC["Tetrahedral_count"][i],
                          DatasetAuNC["Unconnected_triangles_count"][i],
                          DatasetAuNC["Triangles_with_1_shared_vertex_count"][i],
                          DatasetAuNC["Triangles_with_2_shared_vertices_count"][i]]))
    
    x2.append(np.asarray(DatasetAuNC["PersImg"][i]).reshape(100, 100, 1))
    x3.append([DatasetAuNC["Charge"][i]])
    y.append(float(DatasetAuNC[target_variable][i]))

y = np.array(y)
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

'''
Transfer-learning implemented 3-stream model

Transfered model is from AuTM model above
Model is only slightly adjusted from "3-Stream Input Model with EarlyStopping and LOOCV"
'''
predicted_arr = []
true_arr = []
MAE_arr = []
RMSE_arr = []
TotalAccuracy = 0
TotalError = 0
MSE = 0

x1_train = []
x2_train = []
x3_train = []
y_train = []
x1_train_full = []
x2_train_full = []
x3_train_full = []
y_train_full = []
x1_test = []
x2_test = []
x3_test = []
y_test = []



for i in range(len(x1)):
  x1_train_full.append(x1[i])
  x2_train_full.append(x2[i])
  x3_train_full.append(x3[i])
  y_train_full.append(y[i])

for test_index in range(len(x1)):
  model = 0
  print("Cycle: ", test_index)
  x1_train = deepcopy(x1_train_full)
  x2_train = deepcopy(x2_train_full)
  x3_train = deepcopy(x3_train_full)
  y_train = deepcopy(y_train_full)
  # print("Len X train: ", len(X_train))
  # print("Len full: ", len(X_train_full))
  x1_test = list(x1[test_index])
  x2_test = list(x2[test_index])
  x3_test = list(x3[test_index])
  y_test = y[test_index]
  
  x1_train.pop(test_index)
  x2_train.pop(test_index)
  x3_train.pop(test_index)
  y_train.pop(test_index)

  x1_val = []
  x2_val = []
  x3_val = []
  y_val = []

  # taking 10 random datapoints for validation
  random.seed(42)
  random_index = [random.randint(0, 211) for i in range(15)]
  for index in random_index:
    x1_val.append(x1_train[index])
    x1_train.pop(index)
    x2_val.append(x2_train[index])
    x2_train.pop(index)
    x3_val.append(x3_train[index])
    x3_train.pop(index)
    y_val.append(y_train[index])
    y_train.pop(index)

  x1_train = np.asarray(x1_train)
  x2_train = np.asarray(x2_train)
  x3_train = np.asarray(x3_train)
  y_train = np.asarray(y_train)

  x1_val = np.asarray(x1_val)
  x2_val = np.asarray(x2_val)
  x3_val = np.asarray(x3_val)
  y_val = np.asarray(y_val)

  x1_test = np.asarray(x1_test)
  x2_test = np.asarray(x2_test)
  x3_test = np.asarray(x3_test)
  y_test = np.asarray(y_test)

  tf.keras.backend.clear_session()
  # model 1: simplex count and core
  input1 = Input(shape=(5,))
  dense_1 = Dense(5, activation='relu')(input1)
  dense_1_extra = Dense(1, activation='relu')(dense_1)

  # model 2: PI with trained model
  input2 = Input(shape=(100,100,1))
  base_model = load_model("010823_transferTM_dipTotal.h5")
  for layer in base_model.layers[:-2]:
      layer.trainable = False
  new_layer = base_model(input2)
  flatten = Flatten()(new_layer)

  # model 3: Charge
  input3 = Input(shape=(1,))

  # Concatenate models 1,2 and 3
  input = Concatenate()([dense_1_extra, flatten, input3])
  x = Dense(64, activation='relu')(input)
  x = Dense(1)(x)
  model = Model(inputs=[input1, input2, input3], outputs=x)
  model.summary()
  model.compile(
    optimizer = Adam(learning_rate=0.0005),
    loss = 'mean_absolute_error',
    metrics = ['mean_squared_error']
)
  early_stopping = EarlyStopping(monitor='val_loss', patience=40)
  model_checkpoint = ModelCheckpoint(filepath=model_name+'.h5',
                                   save_best_only=True,
                                   save_weights_only=False,
                                   monitor='val_loss',
                                   mode='min',
                                   verbose=1)
  model.fit([x1_train, x2_train, x3_train],
            y_train, 
            batch_size=24, 
            epochs=500,
            verbose = 1,
            validation_data=([x1_val, x2_val, x3_val], y_val),
            callbacks=[early_stopping, model_checkpoint])

  model = load_model(model_name+'.h5')
  
  y_pred = model.predict([np.asarray([x1_test]), np.asarray([x2_test]), np.asarray([x3_test])])
  predicted_arr.append(y_pred)
  true_arr.append(float(y_test))
  abs_error = abs(float(y_pred)- float(y_test))
  TotalError += abs_error
  MSE += float(abs_error**2)

size = len(y)
print(80*"-")
print("Model: 3-stream input CNN with tranfer learning,", target_variable)
print(80*"-")
print("Hyperparameters: Opt: Adam(MAE), lr = 0.001, batchsize=24, Epochs = 500, EarlyS=30, val_split = 15")
print("Structure: 5,5,1 || TF [:-2] || 1 || 64,16,1")
print(80*"-")
print(f"Mean Absolute Error = {TotalError/size}")
print("RMSE =", np.sqrt(MSE/size))
print(80*"-")

MAE_arr.append(TotalError/size)
RMSE_arr.append(np.sqrt(MSE/size))

"""
Plotting predictions vs true values
"""

pred_arr = []
for i in range(len(predicted_arr)):
  pred_arr.append(predicted_arr[i][0][0])
  
plt.figure(figsize=(10,10))
plt.scatter(true_arr, pred_arr, c='crimson', s = 10)

p1 = max(max(pred_arr), max(true_arr))
p2 = min(min(pred_arr), min(true_arr))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
plt.savefig(model_name + ".png", dpi=600)
