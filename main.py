import pandas as pd
df=pd.read_csv('AAPL.csv')
df.head()
df.tail()
df1=df.reset_index()['close']
df1
#stock price from 2015 to 2020
import matplotlib.pyplot as plt
plt.plot(df1)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
df1
#training and testing
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
print(train_data)
#data preprocessing by independent and dependent datas
import numpy
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX),numpy.array(dataY)
#reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step=100
X_train,Y_train=create_dataset(train_data,time_step)
X_test,ytest=create_dataset(test_data,time_step)
print(X_train.shape),print(Y_train.shape)
print(X_test.shape),print(ytest.shape)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
#stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.fit(X_train,Y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
#check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
#transform to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(Y_train,train_predict))
math.sqrt(mean_squared_error(ytest,test_predict))
#plotting
#shift train predictions for plotting
import matplotlib.pyplot as plt
look_back=100
trainPredictPlot=numpy.empty_like(df1)
trainPredictPlot[:, :]= np.nan
trainPredictPlot [look_back:len (train_predict)+look_back, :]= train_predict
#shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :]= numpy.nan
testPredictPlot [len (train_predict)+(look_back*2)+1:len (df1)-1, ] = test_predict
# plot baseline and predictions 
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
x_input=test_data[341:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input
