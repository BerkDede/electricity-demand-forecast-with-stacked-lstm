# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 00:03:23 2022

@author: berk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 14:39:35 2022

@author: Onur Misket
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
import math


df_1 = pd.read_csv('tez3.csv')
df_2 = pd.read_csv('time_stamp.csv')
df_datetime=df_1.iloc[:,0]
df3=df_datetime.str.split(expand=True)
df_date=df3.iloc[:,0]
df_hour=df3.iloc[:,1]
df_hour=df_hour.str.split(pat=':',expand=True)
df_date=df_date.str.split(pat='/',expand=True)
day=df_date.iloc[:,1:2]
day.set_axis(['day'],axis=1,inplace=True)
day=day.astype(int)
day=((day)%7)
df_lag=df_1.iloc[:,1:2].shift(24).fillna(0)
df_lag.rename(columns={'consumption': 'consumption_lag'}, inplace=True)


mon=df_date.iloc[:,0:1]
mon.set_axis(['mounth'],axis=1,inplace=True)
mon=mon.astype(int)
year=df_date.iloc[:,2:3]
year.set_axis(['year'],axis=1,inplace=True)
year=year.astype(int)
hour=df_hour.iloc[:,0:1]
hour.set_axis(['hour'],axis=1,inplace=True)
hour=hour.astype(int)
frames = [df_1, df_2,df_lag,year,day,mon,hour,]
df= pd.concat(frames,axis=1)


df=df.drop('time_stamp',axis=1)

Date_time=df.iloc[:,0:1]
df2=df.set_index(["Date_time"])
Date_time2=[Date_time]
values = df2.values.astype('float32')

remove = ['snowfall']
df_drop= df2[df2.columns.difference(remove)]
values=df_drop[["consumption","precipitation","temperature","irradiance_surface","irradiance_toa",
"snow_mass","cloud_cover","air_density",'consumption_lag',"year","day","mounth","hour"]].values.astype('float32')


df2=pd.DataFrame(df_drop[["consumption","precipitation","temperature","irradiance_surface","irradiance_toa",
"snow_mass","cloud_cover","air_density",'consumption_lag',"year","day","mounth","hour"]])


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an RNN input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y
# Number of lags (hours back) to use for models
lag = 24

# Steps ahead to forecast 
n_ahead = 1

# Share of obs in testing 
test_share = 0.01

ts = df2

nrows = ts.shape[0]


#BURAYI DEĞİŞTİR 4 FARKLI MEVSİMDEN SONUÇ AL
# Spliting into train and test sets
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]

timetest = test.index.to_frame()
timetrain= train.index.to_frame()
time_s=pd.concat([timetest,timetrain])
#scale
scaler = MinMaxScaler(feature_range=(0, 1))

train2= scaler.fit_transform(train)
train2=pd.DataFrame(train2)
values_sc=scaler.fit_transform(values)
#train2=train2.set_index(time_train)


test2= scaler.fit_transform(test)
test2=pd.DataFrame(test2)
#test2=test2.set_index(time_test)
df_for_training_scaled = scaler.fit_transform(df2)
# Creating the final scaled frame 
ts_s = pd.concat([train2, test2])

X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
X_time, Y_time = create_X_Y(time_s.values, lag=lag, n_ahead=n_ahead)
n_ft = X.shape[2]

# Spliting into train and test sets 
Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]
Yval_time=Y_time[int(X.shape[0] * (1 - test_share)):]
print(Ytrain.shape[1])
print(Xtrain.shape[1]) 
print(Xtrain.shape[2])

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of the target data: {Ytrain.shape}")

print(f"Shape of validation data: {Xval.shape}")
print(f"Shape of the validation target data: {Yval.shape}")

# define the Autoencoder model

model = Sequential()
model.add(LSTM(12, kernel_initializer='GlorotNormal', activation='tanh', 
               input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
model.add(LSTM(150, kernel_initializer='GlorotNormal', activation='tanh',
               input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
model.add(LSTM(150,kernel_initializer='GlorotNormal', activation='tanh' , 
               return_sequences=False)) 
model.add(Dense(Ytrain.shape[1],activation='linear'))

model.compile(optimizer='adam', loss='mae',metrics=["mae","mape","mse"])
model.summary()
    
#early stopping
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0, patience=15)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

# fit the model
history = model.fit(Xtrain,Ytrain,
                validation_split=0.2,
                batch_size=256, 
                shuffle=True,
                verbose=2,
                epochs=500,
                callbacks=[es_callback, modelckpt_callback])

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

yhat = model.predict(Xval)
yhat=yhat[:,0:1]
yhat=np.repeat(yhat,repeats= 13 ,axis=1)


yhat_inv = scaler.inverse_transform(yhat)[:,0]

Yval=Yval[:,0:1]
Yval_copies = np.repeat(Yval,repeats= 13 ,axis=1)
y_inv = scaler.inverse_transform(Yval_copies)[:,0]
Yval_time= Yval_time[:,0]


test_mae=mean_absolute_error(y_inv, yhat_inv)
Predict_data=pd.DataFrame(yhat_inv)
Test_data=pd.DataFrame(y_inv)

frames2= [Predict_data,Test_data]
Test=pd.concat(frames2,axis=1)
Test2=Test.astype('float')
Fark=(Test_data)-(Predict_data)
Fark2=pd.DataFrame(Fark)



                     
Predict_data2=Predict_data.astype('float64')
Test_data2=Test_data.astype('float64')
test_mse = math.sqrt(mean_squared_error(Predict_data2,Test_data2))
index_test=pd.DataFrame(index=Yval_time)
frames3= [Predict_data,Test_data]
Test_plot=pd.concat(frames3,axis=1)
Test_plot=Test_plot.astype('float')
Test_plot2 = pd.DataFrame(Test_plot.values,index=Yval_time)

test_mae=mean_absolute_error(Predict_data2,Test_data2)
test_mape=mean_absolute_percentage_error(Predict_data2,Test_data2)
frame_test=[test_mse,test_mae,test_mape]

#Test_plot2=Test_plot2[570:]

plt.figure(figsize=(20,4))
plt.plot(Test_plot2)
plt.xticks(Yval_time[::20],fontsize=10,rotation=60)
plt.ylabel('MWh')
plt.legend(['Pred', 'Real'], loc='upper right')
plt.show()

print(frame_test)