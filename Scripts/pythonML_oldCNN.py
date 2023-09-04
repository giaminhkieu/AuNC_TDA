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
CNN model -- 80/20 split
"""
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def CNNmodel(Dataset, predictor, target_variable, seed):
  X = []
  y = []
  for i in range(len(Dataset)):
    arr = np.array(Dataset[predictor][i])
    X.append(arr.reshape(100,100,1))
  for i in range(len(Dataset)):
    y.append(float(Dataset[target_variable][i]))



  y = np.array(y)
  X = np.array(X)

  Xtrain, Xtest, ytrain, ytest=train_test_split(X, y, test_size=0.2, random_state = seed)


  model = Sequential()
  # model.add(Conv1D(32, 2, activation="relu", input_shape=(10000,1)))
  # model.add(Flatten())

  model.add(Conv2D(64, (11, 11), activation='relu', input_shape=(100,100,1)))
  model.add(MaxPooling2D((2, 2)))
  # model.add(Conv2D(64, (9, 3), activation='relu'))
  # model.add(MaxPooling2D((2, 9)))
  model.add(Flatten())

  model.add(Dense(64, activation="relu"))
  model.add(Dense(16, activation="relu"))
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="Adam")
  model.summary()
  model.fit(Xtrain, ytrain, batch_size=12,epochs=30, verbose=0)

  ypred = model.predict(Xtest)

  print(model.evaluate(Xtrain, ytrain))
  print("MSE: %.4f" % mean_squared_error(ytest, ypred))
  print(np.sqrt(mean_squared_error(ytest, ypred)))



  x_ax = range(len(ypred))
  plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
  plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
  plt.legend()
  plt.savefig('split-results.png')

  totalErr = 0
  MSE = 0
  for i in range(len(ypred)):
    totalErr += abs(ypred[i]-ytest[i])
    MSE += float(abs(ypred[i]-ytest[i]))**2

  print(80*"-")
  print(f"Mean Absolute Error = {totalErr/len(ytest)}")
  print("RMSE =", np.sqrt(MSE/len(ytest)))

  plt.figure(figsize=(10,10))
  plt.scatter(ytest, ypred, c='crimson',alpha = 1, s =10)

  p1 = max(max(ypred), max(ytest))
  p2 = min(min(ypred), min(ytest))
  plt.plot([p1, p2], [p1, p2], 'b-')
  plt.xlabel('True Values', fontsize=15)
  plt.ylabel('Predictions', fontsize=15)
  plt.axis('equal')
  plt.savefig('Mar12_1_mu.png')
  
  return model

CNNmodel(DatasetAuNC,"PersImg","mu",64)
