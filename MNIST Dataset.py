#!/usr/bin/env python
# coding: utf-8

# The core idea behind using deep learning, especially with the MNIST dataset, is to train a model that can accurately predict which number is written in an image, just by "reading" or analyzing the pixel values of that image

# In[1]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


# In[2]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


X_train.shape            #3D Array


# In[4]:


#WATCHING DIMENSION OF FIRST IMAGE
#Pixel is in the range 0-255
X_train[0]


# In[5]:


##WATCHING DIMENSION OF 29th IMAGE
X_train[28]


# In[6]:


y_train


# In[7]:


import matplotlib.pyplot as plt
plt.imshow(X_train[0])


# In[8]:


plt.imshow(X_train[1])


# In[9]:


plt.imshow(X_train[2])


# In[10]:


#Scaling value of X_train and X_test for faster convergence
#Now pixels will come in the range 0-1
X_train = X_train/255
X_test = X_test/255


# In[11]:


X_train


# In[12]:


model = Sequential()

#flattening imput layer: We flatten the data in neural networks while processing images to make it compatible with the input requirements of certain layers in the network, such as fully connected layers(Dense Layer).
#no need to add input nodes while flattening because it will automatically add according to the data
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
#in output layer, we have 10 classes from 0-9
model.add(Dense(10,activation='softmax'))


# In[13]:


model.summary()


# In[14]:


model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])


# In[15]:


history = model.fit(X_train, y_train, epochs = 25, validation_split = 0.2)


# In[16]:


y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis = 1)


# In[17]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[18]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])


# In[19]:


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])


# In[20]:


plt.imshow(X_test[0])


# In[21]:


model.predict(X_test[0].reshape(1,28,28)).argmax(axis = 1)


# In[22]:


plt.imshow(X_test[1])


# In[23]:


model.predict(X_test[1].reshape(1,28,28)).argmax(axis = 1)


# In[ ]:




