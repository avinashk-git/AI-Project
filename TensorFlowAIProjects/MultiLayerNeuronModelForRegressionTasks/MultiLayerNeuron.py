import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#  ********* Read the data set **********************

bike = pd.read_csv('C:\AIProjects\TensorFlowAIProjects\SingleNeuronModel\SampleData\Bike_sharing_daily.csv')

# ******* clean up the data ****************************

bike.describe()
bike.shape
bike=bike.drop(labels=['instant','casual','registered'],axis=1)
bike.dteday = pd.to_datetime(bike.dteday,format='%m/%d/%Y')
bike.index = pd.DatetimeIndex(bike.dteday)
bike=bike.drop(labels=['dteday'],axis=1)

X_cat = bike[['season','yr','mnth','holiday','weekday','workingday','weathersit']]

# *************** Encode the data so that it can be trainable *************************
onehotencoder = OneHotEncoder()
X_cat= onehotencoder.fit_transform(X_cat).toarray()

X_cat=pd.DataFrame(X_cat)
X_numerical=bike[['temp','hum','windspeed','cnt']]

X_numerical=X_numerical.reset_index()
X_all = pd.concat([X_cat,X_numerical],axis=1)


X_all=X_all.drop('dteday',axis=1)

X_all

X=X_all.iloc[:,:-1].values
Y=X_all.iloc[:,-1:].values


scaler = MinMaxScaler()
y=scaler.fit_transform(Y)

# ************ Split the data set into 80% training data and 20 % test data ***************

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


# *********** Create the model **********************

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,activation='relu',input_shape=(35,)))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='linear'))

model.summary()

# *********************** Compile the model ********************
model.compile(optimizer='Adam',loss='mean_squared_error')

# *************** Train the model *********************************
epochs_hist=(model.fit(X_train,Y_train,epochs=50,batch_size=50))

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.title('model loss during progress')
plt.xlabel('Epochs')
plt.ylabel('Training loss')

# ************* Test the Model *******************************

y_predict=model.predict(X_test)

plt.plot(Y_test,y_predict,'^',color='r')
plt.xlabel('Model Prediction')
plt.ylabel('True Values')


