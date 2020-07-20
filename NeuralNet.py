import tensorflow
from tensorflow import keras
import numpy as np
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist

def Dataset_Setup(): #Formats the MNIST dataset to fit the CNN
    (X_Train,Y_Train), (X_Test,Y_Test) = mnist.load_data() #oad MNIST
    
    X_Train = X_Train.reshape(X_Train.shape[0],28,28,1)
    X_Train = X_Train.astype('float32')
    X_Train /= 255
    
    X_Test = X_Test.reshape(X_Test.shape[0],28,28,1)
    X_Test = X_Test.astype('float32')
    X_Test /= 255
    
    Y_Train = keras.utils.to_categorical(Y_Train,10)
    Y_Test = keras.utils.to_categorical(Y_Test,10)

    return(X_Train,X_Test,Y_Train,Y_Test)


def Setup_Model(): #Makes a new untrained CNN model
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1))) #Reformats data
    model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    return(model)

def Model_Summary(model): #Gives a summery of the given model
    model.summary()

def Evaluate_Model(model,X_Test,Y_Test): #Computes and returns the acccuracy of a model
   loss,accuracy = model.evaluate(X_Test,Y_Test,verbose=0)
   return accuracy
    
def Train_Model(model,X_Train,X_Test,Y_Train,Y_Test):
    model.fit(X_Train,Y_Train,batch_size=100,epochs=10,verbose=1,validation_data=(X_Test,Y_Test))
    model.save('trained.h5')
    return(model)
    
def Test_Model(model,test): #Gets predictions from a trained model
    prediction = model.predict(test)
    predictions = []
    for array in prediction: #Reformat prediction array
        for num in array:
            predictions.append(num)

    highest = 0
    array_place = 0
    count = 0
    for num in predictions: #Find the predicted number
        if num > highest:
            highest = num
            array_place = count
        count+=1
    return array_place

def Load_Model(): #Loads a pre-trained model
    model = keras.models.load_model('trained.h5')
    return(model)

def SetupImageData(image): #Prepares given image data to be put through the CNN
    ImageData = np.array([[image]])
    ImageData = ImageData.reshape(ImageData.shape[0],28,28,1)
    ImageData = ImageData.astype('float32')
    ImageData /= 255
    return ImageData
