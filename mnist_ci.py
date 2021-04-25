import sys
import pandas as pd
import numpy as np
from numpy import std
import tensorflow as tf
#from statistics import mean
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
        model.add(tf.keras.layers.Dense(397, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),
            loss=loss,
            metrics=metric
        )
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',min_delta=0,verbose=verbose,patience=5)

        history = model.fit(
            x_train[train_index],
            y_train[train_index],
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_test, y_test),
            callbacks=[early_stop]
            ) 

        val_loss, val_acc = model.evaluate(x_train[test_index], y_train[test_index],verbose=verbose)


        print('|----------------------------------------------------------------------------|')
        fold_number += 1
        print("|  For fold ",(fold_number),"\n|  Loss: ", val_loss, " Accuracy: ",val_acc)
        sum_of_acc += val_acc
        sum_of_loss += val_loss 
        scores.append(val_acc)
        histories.append(history)

    print('|----------------------------------------------------------------------------|')
    print("\n \n The average of the Loss and Accuracy is: \n", "Loss: ",sum_of_loss/fold_number,"\n","Accuracy: ",sum_of_acc/fold_number," \n ") 

    return histories,plot


def prediction(model,x_test,y_test):
    predicts = model.predict(x_test) 
    classes = np.argmax(predicts, axis=1) 



def init_data(train_csv, test_csv,preprocessing_type='normalization'):

    x_train, y_train, x_test, y_test = load_data(train_csv,test_csv) 

    return  data_preprocessing(x_train,y_train, x_test,y_test,preprocessing_type)


x_train, y_train, x_test, y_test = init_data('data/mnist_train.csv','data/mnist_test.csv') 
#model = train_model(x_train,y_train,x_test,y_test,nodes=397,epochs=5,verbose=1,learning_rate=0.001,plot='on')

learning_momentum={0.001:0.2,0.05:0.6,0.1:0.6}
loss_metrics = [ 'categorical_crossentropy','mse']

for i in learning_momentum:
    list_of_histories = list()
    epochs=30
    mean_mse = np.empty((5,epochs))
    mean_ce = np.empty((5,epochs))
    for loss in loss_metrics:
        #means_per_loss = list()
        #nans = np.full((5,epochs),np.nan)
        print("|----------------------------------------------------", "\n|  LOSS METRIC IS: ",loss," ")
        #print('|----------------------------------------------------','\n|  NUM OF NODES: ', node," ")
        print('|----------------------------------------------------','\n|  Learning Rate:', i," Momentum:",learning_momentum[i],"")
        print('|----------------------------------------------------')
    
        history,plot = train_model(x_train,y_train,x_test,y_test,epochs=epochs,verbose=0,learning_rate=i,plot='on',loss=loss,momentum=learning_momentum[i])
        list_of_histories.append(history)
        for hist in range(len(list_of_histories)): 
            means_per_loss = np.full((1,epochs),0)
            for hist_of_loss in range(0,5):
                means_per_loss = np.vstack((means_per_loss,np.asarray(list_of_histories[hist][hist_of_loss].history['loss'] + [np.nan] * (epochs-len( list_of_histories[hist][hist_of_loss].history['loss'] )))))
        
        means_per_loss = np.delete(means_per_loss,(0),axis=0)

        if loss == "mse":
            mean_mse = np.nanmean(means_per_loss,axis=0)
        else: 
            mean_ce = np.nanmean(means_per_loss,axis=0)


    if plot=="on":
        fig = pyplot.figure()
        plots = fig.add_subplot(1, 1, 1)   
        plots.plot([ x for x in range(1,epochs+1)], mean_ce,label="CE")
        plots.plot([ x for x in range(1,epochs+1)], mean_mse, label="MSE")
        plots.set_xlabel("Epochs")
        plots.set_ylabel("Mean Of Loss per Epoch")
        plots.legend()
        pyplot.title("Convergence with Learning Rate= " +  str(i)+" and Momentum="+str(learning_momentum[i]))
        pyplot.show()
        fig.savefig('report/images/'+str(i) + '_' + str(learning_momentum[i]) +'.png')
        
    


       

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
# for loss in loss_metrics:
#    print("|----------------------------------------------------", "\n|  LOSS METRIC IS: ",loss," ")
#    print('|----------------------------------------------------','\n|  NUM OF NODES: ', node," ")
#    print('|----------------------------------------------------')

#    history,plot = train_model(x_train,y_train,x_test,y_test,epochs=epochs,verbose=1,learning_rate=i,plot='off',loss=loss,momentum=learning_momentum[i])
#    list_of_histories.append(history)
# for hist in range(len(list_of_histories)):
#     mean_per_loss = list()
#     for hist_of_loss in range(5):
#         mean_per_loss.append(list_of_histories[hist][hist_of_loss].history['loss'])

#     means.append(mean(mean_per_loss,axis=0))

# if plot=="on":
#     fig = pyplot.figure()
#     plots = fig.add_subplot(1, 1, 1)   
#     plots.plot([ x for x in range(1,epochs+1)], means[0],label="CE")
#     plots.plot([ x for x in range(1,epochs+1)], means[1], label="MSE")
#     plots.set_xlabel("Epochs")
#     plots.set_ylabel("Mean Of Loss per Epoch")
#     plots.legend()
#     pyplot.title("Convergence with Number of Nodes: " +  str(node))
#     pyplot.show()
#     fig.savefig('report/images/%s.png' % str(node))
#