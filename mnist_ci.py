import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from sklearn.preprocessing import OneHotEncoder

#------------------------------------------------# 

# Loading data and splitting into inputs and labels, train and test #
def load_data(train_csv,test_csv):
    train_csv = pd.read_csv(train_csv)
    test_csv = pd.read_csv(test_csv)
    x_train = train_csv.loc[:, train_csv.columns != 'label'].to_numpy().reshape(train_csv.shape[0],28,28)
    y_train = train_csv['label'].to_numpy()
    x_test = test_csv.loc[:, train_csv.columns != 'label'].to_numpy().reshape(test_csv.shape[0],28,28)
    y_test = test_csv['label'].to_numpy() 

    return x_train, y_train, x_test, y_test 


def data_preprocessing(x_train, y_train, x_test, y_test, preprocessing_type='normalization'):
    
    # One hot on labels, because if the model predict a wrong class
    #  we can not say that the input was wrong by an offset, because of different classes
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # y_train = y_train.reshape(60000,10,1)
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    # y_test = y_test.reshape(10000,10,1)

    ######## Normalization ############

    if preprocessing_type == 'normalization':
        x_normalized = tf.keras.utils.normalize(x_train)
        x_test_normalized = tf.keras.utils.normalize(x_test)

        return x_normalized, y_train, x_test_normalized, y_test

    ########## Centering ###################

    elif preprocessing_type == 'centering':
        x_centering = np.zeros((np.shape(x_train)[0], 28, 28))  
        x_test_centering =  np.zeros((np.shape(x_test)[0], 28, 28)) 

        for i,j in zip(range(0,np.shape(x_train)[0]),range(0,np.shape(x_test)[0])):
            scaler = preprocessing.StandardScaler(with_std=False).fit(x_train[i])
            scaler_test = preprocessing.StandardScaler(with_std=False).fit(x_train[j])
            x_centering[i] = scaler.transform(x_train[i])
            x_test_centering[j] = scaler.transform(x_test[j])

        return x_centering, y_train, x_test_centering, y_test 

    ########## Standardize ################

    elif preprocessing_type =='standardize':
        x_standardize = np.zeros((np.shape(x_train)[0], 28, 28))  
        x_test_standardize =  np.zeros((np.shape(x_test)[0], 28, 28)) 

        for i,j in zip(range(np.shape(x_train)[0]),range(np.shape(x_test)[0])): 
            scaler = preprocessing.StandardScaler(with_mean=False).fit(x_train[i])
            scaler_test = preprocessing.StandardScaler(with_std=False).fit(x_train[j])
            x_standardize[i] = scaler.transform(x_train[i])
            x_test_standardize[j] = scaler.transform(x_test[j])


        return x_standardize, y_train, x_test_standardize, y_test

def build_model(n=0.001,loss='sparse_categorical_crossentropy',metric=['accuracy']):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256,input_shape=(48000,), activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=n),
        loss=loss,
        metrics=metric
    )
    return model

def train_model(x_train,y_train,x_test,y_test,epochs=5):

    fold_number = 1
    kfold = KFold(n_splits=5, shuffle=True)
    model = build_model()
    for train_index, test_index in kfold.split(x_train):  

        history = model.fit(
            x_train[train_index],
            y_train[train_index],
            epochs=5,
            verbose=0
            ) 

        val_loss, val_acc = model.evaluate(x_train[test_index], y_train[test_index],verbose=0)
        print('----------------------------------------------------------------------------')
        print("For fold ",fold_number,"\n Loss: ", val_loss, " Accuracy: ",val_acc, " \n")
        fold_number +=1

    return model, val_acc, val_loss


def prediction(model,x_test,y_test):
    predicts = model.predict(x_test) 
    classes = np.argmax(predicts, axis=1) 
    # print("predictions:", classes[0:9], "\n", "expectations:", y_test[0:9]) 
    # if np.array_equal(classes,y_test):
    #     print("predictions are equal with the expected")
    # else: 
    #     print("they are not equal, there were some false predictions")



def init_data(train_csv, test_csv,preprocessing_type='normalization'):

    x_train, y_train, x_test, y_test = load_data(train_csv,test_csv) 

    return  data_preprocessing(x_train,y_train, x_test,y_test,preprocessing_type)

def init_training(learning_rate=1,loss='sparse_categorical_crossentropy',metric=['accuracy']):

    x_train, y_train, x_test, y_test = init_data('data/mnist_train.csv','data/mnist_test.csv') 
    model,val_acc,val_loss = train_model(x_train,y_train, x_test,y_test )
    prediction(model, x_test, y_test)

init_training()