import numpy as np
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


img = np.load('images.npy')
lbl = np.load('labels.npy')
last_train_size = np.load('size.npy')

print("Image size :", img.shape[0])
print("Lables Size: ", lbl.shape[0])
print("Last trained size:", last_train_size)

new_data_size = lbl.size-last_train_size
print("New data found : ", new_data_size)

if new_data_size!=0:
    
    print("Pre-processing data for training")
    img_train, img_test, lbl_train, lbl_test = train_test_split(img, lbl, 
    random_state=0, train_size = .75)
    
    img_train = img_train / 255
    img_test = img_test / 255
    
    img_train = img_train.reshape(img_train.shape[0], img_train.shape[1], img_train.shape[2], 1)
    img_test = img_test.reshape(img_test.shape[0], img_test.shape[1], img_test.shape[2], 1)
    
    img_train = img_train.astype('float32')
    img_test = img_test.astype('float32')
    
    lbl_train = lbl_train.astype('int')
    lbl_test = lbl_test.astype('int')

    print("Train Dataset summery \n ---------------------")
    print("Train data sample size :", img_train.shape[0])
    print("Test data sample size :",img_test.shape[0])

    epochs = input("Enter number of epochs : ")
    batch = input("Enter number of batch size : ")

    epochs = int(epochs)
    batch = int(batch)
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=5, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=5, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(31, activation='softmax'))
    
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
             loss=tensorflow.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    print("Training....")
    model.fit(img_train, lbl_train, epochs=epochs, batch_size=batch, validation_data=(img_test, lbl_test))
    
    score = model.evaluate(img_test, lbl_test)
    print('Loss : ',score[0])
    print('Accuracy :',score[1])
    print(model.summary()) 
    
    model.save('../model.h5')
    print("Model succesfully updated with new data")

    save_size = lbl.shape[0]
    np.save('../data/size', save_size)
    
else:
    print("No new data to update the model, program will exit now")






