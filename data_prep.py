import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold

# LOADING DATA AND SPLITTING IT.

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

#for i in range(0,60000):
#    # CENTRALIZATION
#    scaler = preprocessing.StandardScaler(with_std=False).fit(x_train[i])
#    x_centralized[i] = scaler.transform(x_train[i])
#    # STANDARDIZATION
#    scaler = preprocessing.StandardScaler(with_mean=False).fit(x_train[i])
#    x_standardized[i] = scaler.transform(x_train[i])
#


kfold = KFold(n_splits=5, shuffle=False)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

for train_index, test_index in kfold.split(x_normalized, y_train):
    x = x_normalized[train_index]
    y = y_train[train_index]
    model.fit(x, y, epochs=5)

val_loss, val_acc = model.evaluate(x_test_normalized, y_test)
print(val_loss, val_acc)
predicts = model.predict(x_test_normalized)

classes = np.argmax(predicts, axis=1)

print("predictions:", classes[0:9], "\n", "expectations:", y_test[0:9]) 