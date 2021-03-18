import pandas as pd 
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

train_csv = pd.read_csv("data/mnist_train.csv")
test_csv = pd.read_csv("data/mnist_test.csv")

x_train = train_csv.loc[:, train_csv.columns != 'label'].to_numpy()
y_train = train_csv['label'].to_numpy()

x_test = test_csv.loc[:, train_csv.columns != 'label'].to_numpy()
y_test = test_csv['label'].to_numpy()

x_train=x_train.reshape(60000,28,28)
x_test=x_test.reshape(10000,28,28)

print(y_test.reshape(10000,28))

