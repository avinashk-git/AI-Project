import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp_df=pd.read_csv('C:\AIProjects\TensorFlowAIProjects\SingleNeuronModel\SampleData\Celsius+to+Fahrenheit.csv')

# ********************  Optional steps to validate the data starts here ***************
# temp_df.head(5)
# temp_df.tail(10)
# temp_df.describe()
# temp_df.info()

# sns.scatterplot(x=temp_df['Celsius'],y=temp_df['Fahrenheit'])
# temp_df['Celsius'].shape
# temp_df['Fahrenheit'].shape

#  ************** Data validation ends here **************************

# **************** Initiate the model  ****************************

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,input_shape=[1]))
# model.summary()

# *************  Compile the model and train it also set the number of Epochs ****************
model.compile(optimizer=tf.keras.optimizers.Adam(0.5),loss='mean_squared_error')
epochs_hist = model.fit(temp_df['Celsius'],temp_df['Fahrenheit'],epochs=500)
epochs_hist.history.keys()

# *********** Optional step to plot the training loss *********************
plt.plot(epochs_hist.history['loss'])
plt.title('model loss progress during training')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

model_weights=model.get_weights()
print(f' weights of the model after training  {model_weights}')

# ********************* Validate the model ******************8
Temp_c=0
Temp_f=model.predict([Temp_c])
print(f'Equivalent Fahrenheit temperature {Temp_f}')


