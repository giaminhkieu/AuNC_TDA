from ElementsInfo import *
import os
import numpy as np
from PersistentImageCode import *
from VariancePersistCode import *
from GenerateImagePI import *
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

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


"""
CNN model: PI only input, with LOOCV
"""
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from copy import deepcopy

MAE_arr = []
RMSE_arr = []

X = []
X2 = []
y = []
y2 = []

target_variable = "u298"
model_name = "020823_PIonly_" + target_variable

for i in range(len(DatasetAuNC)):
  current = np.array(DatasetAuNC["PersImg"][i])
  X.append(current.reshape(100,100,1))

for i in range(len(DatasetAuNC)):
  y.append(DatasetAuNC[target_variable][i])


y = np.array(y)
X = np.array(X)


predicted_arr = []
true_arr = []
TotalAccuracy = 0
TotalError = 0
MSE = 0

X_train = []
y_train = []
X_train_full = []
y_train_full = []
X_test = []
y_test = []

for i in range(len(X)):
  X_train_full.append(X[i])
  y_train_full.append(y[i])


for test_index in range(len(X)):
  model = 0
  print("Cycle: ", test_index)
  X_train = deepcopy(X_train_full)
  y_train = deepcopy(y_train_full)

  X_test = list(X[test_index])
  y_test = y[test_index]
  
  X_train.pop(test_index)
  y_train.pop(test_index)
  
  X_val = []
  y_val = []
  
  random.seed(42)
  random_index = [random.randint(0,211) for i in range(15)]  
  for index in random_index:
    X_val.append(X_train[index])
    X_train.pop(index)
    y_val.append(y_train[index])
    y_train.pop(index)
  
  X_train = np.asarray(X_train)
  X_val = np.asarray(X_val)
  y_train = np.asarray(y_train)
  y_val = np.asarray(y_val)
  X_test = np.asarray(X_test)
  y_test = np.asarray(y_test)

  tf.keras.backend.clear_session()

  input2 = Input(shape=(100,100,1))
  conv_1 = Conv2D(16, (3, 3), activation='relu')(input2)
  maxpool_1 = MaxPooling2D((2, 2))(conv_1)
  conv_2 = Conv2D(16, (3, 3), activation='relu')(maxpool_1)
  maxpool2 = MaxPooling2D((2,2))(conv_2)
  flatten = Flatten()(maxpool2)
  dense_2 = Dense(64, activation='relu')(flatten)
  dense_3 = Dense(16, activation='relu')(dense_2)
  dense_4 = Dense(1)(dense_3)

  model = Model(inputs = [input2], outputs=dense_4)

  # optimizer = keras.optimizers.Adam(lr=0.001)
  model.compile(loss="mae", optimizer=Adam(learning_rate=0.0005), metrics = ['mean_squared_error'])
  model.summary()

  early_stopping = EarlyStopping(monitor='val_loss', patience=40)
  model_checkpoint = ModelCheckpoint(filepath=model_name+'.h5',
                                   save_best_only=True,
                                   save_weights_only=False,
                                   monitor='val_loss',
                                   mode='min',
                                   verbose=1)

  model.fit(X_train, y_train, batch_size=24, epochs=500, verbose=1, validation_data = ([X_val],y_val), callbacks = [early_stopping, model_checkpoint])

  model = load_model(model_name+'.h5')

  y_pred = model.predict(np.asarray([X_test]))
  predicted_arr.append(y_pred)
  true_arr.append(float(y_test))
  abs_error = abs(float(y_pred)- float(y_test))
  TotalError += abs_error
  MSE += float(abs_error**2)
  # print(model.evaluate(X_train, y_train))
  # print("MSE: %.4f" % mean_squared_error(y_test, y_pred))
  # print(np.sqrt(mean_squared_error(y_test, y_pred)))



# x_ax = range(len(y_pred))
# plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
# plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show()

size = len(y)
print(80*"-")
print("Model: CNN PI only, LOOCV,", target_variable)
print(80*"-")
print("Hyperparameters:")
print("Conv(16,3,3), Maxpool(2,2), Conv(16,3,3), Maxpool(2,2), flatten, 32,16, 1")
print("Opt: Adam, MAE loss, lr= 0.0005, batch_size = 24, epochs = 500 - ES 40")
print(80*"-")
print(f"Mean Absolute Error = {TotalError/size}")
print("RMSE =", np.sqrt(MSE/size))
print(80*"-")

MAE_arr.append(TotalError/size)
RMSE_arr.append(np.sqrt(MSE/size))

plt.figure(figsize=(10,10))
plt.scatter(true_arr, predicted_arr, c='crimson', s = 10)

p1 = max(max(predicted_arr), max(true_arr))
p2 = min(min(predicted_arr), min(true_arr))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
plt.savefig(model_name+".png", dpi=600)

