import pandas as pd
import numpy as np
import tensorflow as tf


train_csv = pd.read_csv("data/mnist_train.csv")
test_csv = pd.read_csv("data/mnist_test.csv")


x_train = train_csv.loc[:, train_csv.columns != 'label'].to_numpy()
y_train = train_csv['label'].to_numpy()

x_test = test_csv.loc[:, train_csv.columns != 'label'].to_numpy()
y_test = test_csv['label'].to_numpy()

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

# CENTRALIZATION
x_centralized = np.zeros((60000, 28, 28))
x_standardized = np.zeros((60000, 28, 28))

# NORMALIZATION
x_normalized = tf.keras.utils.normalize(x_train)
x_test_normalized = tf.keras.utils.normalize(x_test)

# print(x_train[0])
print(type(x_train))
# print(train_csv.values[ 0 ])


#for i in range(0,60000):
#    # CENTRALIZATION
#    scaler = preprocessing.StandardScaler(with_std=False).fit(x_train[i])
#    x_centralized[i] = scaler.transform(x_train[i])
#    # STANDARDIZATION
#    scaler = preprocessing.StandardScaler(with_mean=False).fit(x_train[i])
#    x_standardized[i] = scaler.transform(x_train[i])
#