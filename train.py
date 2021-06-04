#!/usr/bin/env python
# coding: utf-8

# In[37]:


get_ipython().system('pip install opencv.python')


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# In[39]:


DATADIR = "C:/Users/Aliffia Nuraini/labelImg-master/KK"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break


# In[40]:


print(img_array)


# In[41]:


print(img_array.shape)


# In[43]:


IMG_SIZE = 28

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = "gray")
plt.show()


# In[44]:


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()


# In[45]:


print(len(training_data))


# In[46]:


import random

random.shuffle(training_data)


# In[47]:


for sample in training_data[:10]:
    print(sample[1])


# In[48]:


x = []
y = []


# In[49]:


for features, label in training_data:
    x.append(features)
    y.append(label)
    
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[50]:


import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[51]:


pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)


# In[52]:


x[1]

