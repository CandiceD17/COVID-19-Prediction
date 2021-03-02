# import torch
# from torch import nn
# import numpy as np  
# import pandas as pd 
# import matplotlib.pyplot as plt  

# df = pd.read_csv('ucla2020-cs145-covid19-prediction/train.csv')

import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt 
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
import keras
from scipy.optimize import curve_fit

data = pd.read_csv('ucla2020-cs145-covid19-prediction/train.csv')
data = data.loc[data['Province_State']=='California'].filter(items=['Province_State', 'Date', 'Confirmed','Deaths'])
print (data.shape)

t_data='2020-01-21'
num=1
sc=0
sd=0
cum_data=[]
for index, row in data.iterrows():  # get cumulative data
    sc=sc+row["Confirmed"]
    sd=sd+row["Deaths"]
    if t_data!=row["Date"]:
        t_data=row["Date"]
        cum_data.append((num,sc,sd))
        num=num+1
        sc=0
        sd=0

print(num)

sin_data=[]
sin_data.append(cum_data[0])
for i in range(len(cum_data)-1):
    sin_data.append((cum_data[i+1][0],cum_data[i+1][1]-cum_data[i][1],cum_data[i+1][2]-cum_data[i][2]))
cum_data = pd.DataFrame(cum_data,columns=['Day','Confirmed','Deaths'])    
cum_data.to_csv("cum_data.csv",index=False)
sin_data = pd.DataFrame(sin_data,columns=['Day','Confirmed','Deaths'])    
sin_data.to_csv("sin_data.csv",index=False)    

# validation
def create_dataset(dataset,step,validate_rate=0.67):   
    dataX,dataY=[],[]
    trainX,trainY,testX,testY=[],[],[],[]
    for i in range(len(dataset)-step):
        a = dataset[i:(i+step), 0]
        dataX.append(a)
        dataY.append(dataset[i+step, 0])
    trainX=dataX[:int(validate_rate*len(dataX))]
    trainY=dataY[:int(validate_rate*len(dataY))]
    testX=dataX[int(validate_rate*len(dataX)):]
    testY=dataY[int(validate_rate*len(dataY)):]
    print (np.asarray(dataX).shape)
    return np.array(trainX), np.array(trainY),np.array(testX),np.array(testY)

# loss function
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()  
def loss_fn(y_true, y_pred):
    return [abs(a-b)/abs(b) for a,b in zip(y_pred, y_true)]

#split data
dataset=cum_data["Confirmed"].to_frame()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train = dataset
step = 40
validate_rate=0.75
trainX, trainY, testX, testY = create_dataset(train,step,validate_rate)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
history = LossHistory()
model = Sequential()
model.add(LSTM(100, input_shape=(1, step)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, validation_split=0.1,batch_size=32, verbose=1,callbacks=[history])
history.loss_plot("batch")

testY = scaler.inverse_transform([testY])
trainPredict=model.predict(trainX)
a=np.append(scaler.inverse_transform(trainX[0]),scaler.inverse_transform(trainPredict))

temp=testX[0].reshape(1,testX.shape[1],testX.shape[2])
result=[]
for i in range(len(testY[0])):
    pre=model.predict(temp)[0][0]
    result.append(pre)
    temp=temp[0][0][1:]
    temp=np.append(temp,pre)
    temp=temp.reshape(1,1,temp.shape[0])
result=np.array(result)
result=result.reshape(np.array(result).shape[0],1)
predict=scaler.inverse_transform(result)    

print (len(testY[0])) # test


plt.plot(scaler.inverse_transform(dataset),label='true')
plt.plot(a,label='trainpredict')
plt.plot(range(143-len(predict),143),predict,label='testpredict')
plt.title('Confirmed')
plt.legend()
plt.show()

plt.plot(range(0,24),scaler.inverse_transform(dataset)[144-len(predict):144],label='true')
plt.plot(range(0,26),predict,label='testpredict')
plt.legend()
plt.title('Confirmed')
plt.show()
def mape(pred, true):
    length = len(pred)
    res = 0
    for i in range(length):
        res += np.abs(pred[i] - true[i]) / true[i]
    return res / length

TestScore = mape(predict, testY[0,:])
print('Test Score: %.2f MAPE' % (TestScore))
