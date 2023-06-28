import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('dataset.csv')
df.head()
df.drop(columns=df.columns[-1],  axis=1,  inplace=True)
df.isnull().sum()
x = df.iloc[:,0:132] # iloc[:,:]
x.head()
y = pd.get_dummies(df.iloc[:,132:])
y.head()
# Split the training and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=12)

# Build an ANN model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ANN Model

model = Sequential()
model.add(Dense(5,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(41,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=20,batch_size=10,validation_data=(xtest,ytest))
ypred = model.predict(xtest)
ypred=(ypred == ypred.max(axis=1)[:,None]).astype(int)
ypred[:10,:]
model.save('pred_model1.h5')