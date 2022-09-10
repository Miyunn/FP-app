
print("Importing libraries")
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

print("Reading folders and creating classes")

path = "dataset/train/"
path_test = "dataset/test/"
files = os.listdir(path)[:31]
files_test = os.listdir(path_test)[:31]
print("Detected train classes : ",files)
print("Detected test classes : ",files_test)


classes={'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9, '11':10, '12':11, '25':12, '38':13, '51':14,
        '64':15, '77':16, '90':17, '93':18, '105':19, '120':20, '134':21, '149':22, '164':23, '179':24, '190':25, '198':26,
        '208':27, '250':28, '264':29, '274':30}


print("Creating train data list")
img_train=[]
lbl_train=[]

for cl in classes:
    pth = path+cl
    for img_name in os.listdir(pth):
        img = cv2.imread(pth+"/"+img_name,0)
        img_train.append(img)
        lbl_train.append(classes[cl])
print("Train data list imported")  

print("Creating test data list")
img_test=[]
lbl_test=[]

for cl in classes:
    pth = path_test+cl
    for img_name in os.listdir(pth):
        img = cv2.imread(pth+"/"+img_name,0)
        img_test.append(img)
        lbl_test.append(classes[cl])
print("Test Data imported")


print("Converting list to array type")

img_train = np.array(img_train)
lbl_train = np.array(lbl_train)
print("Train data type")
print(type(img_train))
print(type(lbl_train))

img_test = np.array(img_test)
lbl_test = np.array(lbl_test)

print("Train data type")
print(type(img_test))
print(type(lbl_test))


#plt.imshow(img_train[257], cmap='gray')
#print(lbl_train[257])


print("Reshapping Train and Test data")
img_train = img_train.reshape(img_train.shape[0], img_train.shape[1], img_train.shape[2], 1)
img_test = img_test.reshape(img_test.shape[0], img_test.shape[1], img_test.shape[2], 1)


print("Train and test data shapes")
print(img_train.shape)
print(img_test.shape)


print("Changing lables to categorical")
lbl_train = to_categorical(lbl_train, 31)
lbl_test = to_categorical(lbl_test, 31)

#print(np.max(img_train[0]))
#print(np.min(img_train[0]))

img_train = img_train / 255
img_test = img_test / 255

#print(np.max(img_train[0]))
#print(np.min(img_train[0]))

#print(type(img_train[0][0][0][0]))
#print(type(img_test[0][0][0][0]))

#print("Convering to type to float32")
img_train = img_train.astype('float32')
img_test = img_test.astype('float32')


print("Dimentions of train data")
print(type(img_train[0][0][0][0]))


model = Sequential()
model.add(Conv2D(256, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(1024, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(31, activation='softmax'))

print("Model loaded")


model.compile(optimizer=tensorflow.keras.optimizers.Adadelta(learning_rate=0.1),
             loss=tensorflow.keras.losses.categorical_crossentropy, metrics=['accuracy'])


print("Training....")
model.fit(img_train, lbl_train, epochs=25, batch_size=10, validation_data=(img_test, lbl_test))


score = model.evaluate(img_test, lbl_test)


print('Test Loss : ', score[0])
print('Test Accuracy :',score[1])



model.save('test/model.h5')
print("Model saved")





