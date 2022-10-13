import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# In[2]:


print("Importing data from directory...")

path = "dataset/train/"
path_test = "dataset/test/"
files = os.listdir(path)[:31]
files_test = os.listdir(path_test)[:31]

classes={'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9, '11':10, '12':11, '25':12, '38':13, '51':14,
        '64':15, '77':16, '90':17, '93':18, '105':19, '120':20, '134':21, '149':22, '164':23, '179':24, '190':25, '198':26,
        '208':27, '250':28, '264':29, '274':30 }


# In[3]:


print("Labeling train data...")
img_train=[]
lbl_train=[]

for cl in classes:
    pth = path+cl
    for img_name in os.listdir(pth):
        img = cv2.imread(pth+"/"+img_name,0)
        img_train.append(img)
        lbl_train.append(classes[cl])


# In[4]:


print("Labeling test data...")
img_test=[]
lbl_test=[]

for cl in classes:
    pth = path_test+cl
    for img_name in os.listdir(pth):
        img = cv2.imread(pth+"/"+img_name,0)
        img_test.append(img)
        lbl_test.append(classes[cl])


# In[5]:


print("converting to np array")

img_train = np.array(img_train)
lbl_train = np.array(lbl_train)

img_test = np.array(img_test)
lbl_test = np.array(lbl_test)

save_size = lbl_train.size

print(save_size)


# In[6]:


np.save('../data/images', img_train)
np.save('../data/labels', lbl_train)
np.save('../data/size', save_size)
print("Saved for updating")


# In[7]:


plt.imshow(img_train[1], cmap='gray')
print(lbl_train[1])


# In[8]:


print("Preprocessing data for training....")


# In[9]:


img_train = img_train.reshape(img_train.shape[0], img_train.shape[1], img_train.shape[2], 1)
img_test = img_test.reshape(img_test.shape[0], img_test.shape[1], img_test.shape[2], 1)


# In[10]:


img_train = img_train / 255
img_test = img_test / 255


# In[11]:


img_train = img_train.astype('float32')
img_test = img_test.astype('float32')


# In[12]:


print("Train shape", img_train.shape)
print("Test shape", img_test.shape)


# In[13]:


epochs = 50
batch = 32

print("Epochs : ",epochs)
print("Batch size : ",batch)


# In[14]:


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
             loss=tensorflow.keras.losses.sparse_categorical_crossentropy, 
             metrics=['accuracy'])

print("Training....")
model.fit(img_train, lbl_train, epochs=epochs, batch_size=batch, 
        validation_data=(img_test, lbl_test))


score = model.evaluate(img_test, lbl_test)

print(model.summary()) 
print('Loss : ', score[0])
print('Accuracy :',score[1])


model.save('../model.h5')
print("Model saved")


# In[ ]:




