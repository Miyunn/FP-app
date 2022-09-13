#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter.ttk import *
import PIL.ImageGrab
from PIL import Image
from keras.models import load_model
import numpy as np

# load model
model = load_model('model.h5')


operator = "Prediction: "
cls = ""

# clear canvas function
def Clear():
    cv.delete("all")
    global operator2
    text_input.set(cls)

# image extraction and predict function
def Predict():
    file = 'Data\image.jpg'
    
    if file:
        # save the canvas in jpg format
        x = root.winfo_rootx() + cv.winfo_x()
        y = root.winfo_rooty() + cv.winfo_y()
        x1 = x + cv.winfo_width()
        y1 = y + cv.winfo_height()
        PIL.ImageGrab.grab().crop((x,y,x1,y1)).save(file)
        
        img = Image.open(file).convert("L")
        img = img.resize((80,80))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,80,80,1)
        
        #predicting
        y_pred2 = model.predict(im2arr)
        classes_x=np.argmax(y_pred2,axis=1)
        #covert class to scalar
        x = classes_x[0]
        
        #setoutput
        global operator
        operator = operator+str(x)
        text_input.set(operator)
        operator = operator = "Prediction: "

# canvas draw function
def paint(event):
    old_x = event.x
    old_y = event.y        
        
    cv.create_line(old_x, old_y, event.x, event.y,
                               width=10, fill="white",
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

# Tkinter Interface
root = Tk()
root.title("Sinhala Model Test")


text_input = StringVar()


textdisplay = Entry(root, 
               textvariable = text_input,  
               justify = 'center')



btn1 = Button(root, text = "Predict", command = lambda:Predict())
btn2 = Button(root, text = "Erase", command = lambda:Clear())

cv = Canvas(root,width=400,height=400,bg="black",)
cv.bind('<B1-Motion>', paint) 


cv.grid(row = 0, column = 0)
textdisplay.grid(row = 0, column = 1)
btn1.grid(row = 1, column = 0)
btn2.grid(row = 0, column = 2)


root.mainloop()


# In[ ]:





# In[ ]:




