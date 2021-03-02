from datetime import datetime
import numpy as np             
import pandas as pd      
import matplotlib.pylab as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf


from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

Data = pd.read_csv('ucla2020-cs145-covid19-prediction/train.csv', index_col=0)


Data = Data.loc[Data['Province_State']=='California'].filter(items=['Date', 'Confirmed'])
# Data = Data.reset_index()

Data['Date']= pd.to_datetime(Data['Date'],infer_datetime_format=True) 
indexedDataset = Data.set_index(['Date'])
indexedDataset = indexedDataset.iloc[:-1]


plt.figure(figsize=(15,5),dpi = 80)
plt.xticks(rotation=45)


plt.xlabel('Date')
plt.ylabel('confirmedCount')
plt.plot(indexedDataset)
plt.show()

indexedDataset_logScale = np.log(indexedDataset)

##Differencing
datasetLogDiffShifting_1 = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting_1)
datasetLogDiffShifting_2 = datasetLogDiffShifting_1 - datasetLogDiffShifting_1.shift()
plt.plot(datasetLogDiffShifting_2)
datasetLogDiffShifting_3 = datasetLogDiffShifting_2 - datasetLogDiffShifting_2.shift()
plt.plot(datasetLogDiffShifting_3)
plt.show()

model_3 = ARIMA(indexedDataset_logScale, order=(3,1,3))
results_ARIMA = model_3.fit(disp=-1)
plt.plot(datasetLogDiffShifting_1)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting_1['Confirmed'])**2))
print('Plotting ARIMA model')
plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(indexedDataset_logScale['Confirmed'].iloc[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)


# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
# predictions_ARIMA

plt.xticks(rotation=45)
plt.plot(indexedDataset, label = "True Data")
plt.plot(predictions_ARIMA, label = "Pred Data")
plt.legend()
plt.show()


print (indexedDataset['Confirmed'].iloc[0])

mape = 0
# for i in range (len(predictions_ARIMA.array)):
#     mape += abs((indexedDataset.array[i] - predictions_ARIMA.array[i])/indexedDataset.array[i])

# print (predictions_ARIMA.array)
