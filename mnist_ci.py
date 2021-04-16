import sys
import pandas as pd
import numpy as np
from numpy import std
from numpy import mean 
import tensorflow as tf
from statistics import mean
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from matplotlib import pyplot 

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
    # we can not say that the input was wrong by an offset, because of different classes
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)

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

def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()

def train_model(x_train,y_train,x_test,y_test,epochs=5,nodes=10,verbose=0,loss='categorical_crossentropy',metric=['accuracy'],learning_rate=0.001,momentum=0,plot='off'):

    fold_number = 0
    scores ,histories = list(), list()
    sum_of_loss = sum_of_acc =  0
    kfold = KFold(n_splits=5, shuffle=False)


    # K-Fold iterations
    for train_index, test_index in kfold.split(x_train,y_train):  

        # Model configuration
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten()) # Change input from shape (,28,28) to (,784)
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),
            loss=loss,
            metrics=metric
        )

        history = model.fit(
            x_train[train_index],
            y_train[train_index],
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_train[test_index], y_train[test_index])
            ) 

        val_loss, val_acc = model.evaluate(x_train[test_index], y_train[test_index],verbose=0)


        print('|----------------------------------------------------------------------------|')
        fold_number += 1
        print("|  For fold ",(fold_number),"\n|  Loss: ", val_loss, " Accuracy: ",val_acc)
        sum_of_acc += val_acc
        sum_of_loss += val_loss 
        scores.append(val_acc)
        histories.append(history)
    print('|----------------------------------------------------------------------------|')
    print("\n \n The average of the Loss and Accuracy is: \n", "Loss: ",sum_of_loss/fold_number,"\n","Accuracy: ",sum_of_acc/fold_number," \n ") 
    if(plot == 'on'):
        summarize_diagnostics(histories)
        # summarize estimated performance
        summarize_performance(scores)

    return model 


def prediction(model,x_test,y_test):
    predicts = model.predict(x_test) 
    classes = np.argmax(predicts, axis=1) 



def init_data(train_csv, test_csv,preprocessing_type='normalization'):

    x_train, y_train, x_test, y_test = load_data(train_csv,test_csv) 

    return  data_preprocessing(x_train,y_train, x_test,y_test,preprocessing_type)


x_train, y_train, x_test, y_test = init_data('data/mnist_train.csv','data/mnist_test.csv') 
#model = train_model(x_train,y_train,x_test,y_test,nodes=397,epochs=20,verbose=1,learning_rate=0.001,plot='on')

hidden_nodes = [10,397,794]
loss_metrics = [ 'categorical_crossentropy','mse']

for node in hidden_nodes:
    for loss in loss_metrics:
        print("|----------------------------------------------------", "\n|  LOSS METRIC IS: ",loss," ")
        print('|----------------------------------------------------','\n|  NUM OF NODES: ', node," ")
        print('|----------------------------------------------------')

        model = train_model(x_train,y_train,x_test,y_test,nodes=node,epochs=20,verbose=0,learning_rate=0.001,plot='off',loss=loss)
        

#prediction(model, x_test, y_test)



#             ----------------------------------- Probably to be used -------------------------------------------------
#             ---------------------------------------------------------------------------------------------------------
#             ---     print("predictions:", classes[0:9], "\n", "expectations:", y_test[0:9])                       ---
#             ---     if np.array_equal(classes,y_test):                                                            ---
#             ---         print("predictions are equal with the expected")                                          ---
#             ---     else:                                                                                         ---
#             ---         print("they are not equal, there were some false predictions")                            --- 
#             ---        print(mean( history.history['accuracy'] ))                                                 ---
#             ---        plt.plot(history.history['loss'])                                                          ---
#             ---        plt.plot(history.history['val_loss'])                                                      ---
#             ---        plt.title('model loss'                                                                     ---
#             ---        plt.ylabel('loss')                                                                         ---
#             ---        plt.xlabel('epoch')                                                                        ---
#             ---        plt.legend(['train', 'test'], loc='upper left'                                             ---
#             ---        plt.show()                                                                                 ---
#             ---------------------------------------------------------------------------------------------------------
#             ---------------------------------------------------------------------------------------------------------