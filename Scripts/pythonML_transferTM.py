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
Initializations and data appending for model below
'''
print("Initializing...")

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
x2, y = [], []
target_variable = "dipTotal"
model_name = "010823_transferTM_" + target_variable


for i in range(len(DatasetTM)):
    
    x2.append(np.asarray(DatasetTM["PersImg"][i]).reshape(100, 100, 1))
    y.append(float(DatasetTM[target_variable][i]))

y = np.array(y)
x2 = np.array(x2)

'''
PI only CNN model with EarlyStopping implemented, No LOOCV - Training on AuTM dataset for transfer learning
'''
predicted_arr = []
true_arr = []
MAE_arr = []
RMSE_arr = []
TotalAccuracy = 0
TotalError = 0
MSE = 0

x2_train = []
y_train = []
x2_train_full = []
y_train_full = []
x2_test = []
y_test = []

#Train-Validation split
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=42)
tf.keras.backend.clear_session()

print("Running model...")
# Model structure
# input2 = Input(shape=(16,16,1))
# Resnet = resnet18(input2) #resnet18 replacement block
input2 = Input(shape=(100,100,1))
conv_1 = Conv2D(16, (3, 3), activation='relu')(input2)
maxpool_1 = MaxPooling2D((2, 2))(conv_1)
conv_2 = Conv2D(16, (3, 3), activation='relu')(maxpool_1)
maxpool_2 = MaxPooling2D((2,2))(conv_2)
# conv_3 = Conv2D(128, (3, 3))(maxpool_2)
# maxpool_3 = MaxPooling2D((2,2))(conv_3)
flatten = Flatten()(maxpool_2)
dense_2 = Dense(32, activation='relu', name="dense_a")(flatten)
dense_3 = Dense(16, activation='relu', name="dense_b")(dense_2)
dense_4 = Dense(1, name="dense_c")(dense_3)

# Compiling the model
model = Model(inputs= input2, outputs=dense_4)
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

model.fit([x2_train],
          y_train, 
          batch_size=128, 
          epochs=500,
          verbose = 1,
          validation_data=([x2_test], y_test),
          callbacks=[early_stopping, model_checkpoint])

print("fitting...")
model = load_model(model_name+'.h5')

# Predicting with the best model
for i in range(len(x2_test)):
  y_pred = model.predict(np.asarray([x2_test[i]]))
  predicted_arr.append(y_pred)
  true_arr.append(float(y_test[i]))
  abs_error = abs(float(y_pred)- float(y_test[i]))
  TotalError += abs_error
  MSE += float(abs_error**2)


test_score = model.evaluate(x2_test, y_test, verbose=0)


size = len(y_test)
print(80*"-")
print("Model: PI training on AuTM,", target_variable)
print("Structure: Conv(16,3,3), Maxpool(2,2), Conv(16,3,3), MP(2,2), Flatten, 32, 16, 1")
print("Hyperparams: Adam MSE loss, lr = 0.0005, batchsize = 128, Epochs = 500 (earlyStopping patience = 40, 80-20 split")
print(80*"-")
print(f"Mean Absolute Error = {TotalError/size}")
print("RMSE =", np.sqrt(MSE/size))
print(80*"-")

MAE_arr.append(TotalError/size)
RMSE_arr.append(np.sqrt(MSE/size))

print("Test score of Keras: ", test_score)
pred_arr = []
for i in range(len(predicted_arr)):
  pred_arr.append(predicted_arr[i][0][0])


plt.figure(figsize=(10,10))
plt.scatter(true_arr, pred_arr, c='cyan', s=10)
p1 = max(max(pred_arr), max(true_arr))
p2 = min(min(pred_arr), min(true_arr))
plt.plot([p1,p2], [p1,p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
plt.savefig(model_name+".png", dpi=600)
