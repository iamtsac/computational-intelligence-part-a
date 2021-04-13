import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# Imports
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np


train_csv = pd.read_csv('data/mnist_train.csv')
test_csv = pd.read_csv('data/mnist_test.csv')
X_train = train_csv.loc[:, train_csv.columns !=
                        'label'].to_numpy().reshape(train_csv.shape[0], 784)
y_train = train_csv['label']
X_test = test_csv.loc[:, train_csv.columns !=
                      'label'].to_numpy().reshape(test_csv.shape[0], 784)
y_test = test_csv['label']

# Configuration options
no_classes = len(np.unique(y_train))
img_width, img_height = 28, 28
no_epochs = 25
verbosity = 1
batch_size = 250

# Reshape data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test =  X_test.reshape(X_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Convert targets into one-hot encoded format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Normalize the data
X_train = normalize(X_train)
X_test = normalize(X_test)

# Create the model

kfold = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kfold.split(X_train,y_train):  
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
    
    # Fit data to model
    model.fit(X_train[train_index], y_train[train_index],
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity)
           
    val_loss, val_acc = model.evaluate(x_train[test_index], y_train[test_index],verbose=0)