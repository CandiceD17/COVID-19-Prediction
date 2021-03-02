from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_total = pd.read_csv('ucla2020-cs145-covid19-prediction/train.csv')

states = df_total['Province_State'].unique()


deg_confirm = [4, 1, 3, 4, 1, 2, 1, 2, 2, 4, 3, 3, 2, 2, 4, 3, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 2, 2, 3, 2, 2, 1, 4, 4, 3, 4, 1, 1, 2, 2, 2, 2, 1]
deg_death = [2, 4, 3, 2, 3, 4, 2, 1, 3, 4, 3, 3, 2, 4, 2, 3, 2, 2, 1, 4, 4, 2, 3, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 3, 2, 3, 1, 3, 2, 2, 3, 2, 2, 1, 1, 4, 2, 3, 1]


itv_confirm = [40, 40, 40, 71, 71, 40, 71, 71, 40, 71, 40, 40, 40, 40, 71, 40, 40, 40, 40, 40, 40, 40, 40, 71, 40, 40, 40, 40, 40, 40, 40, 40, 40, 71, 40, 40, 40, 40, 40, 40, 142, 40, 71, 40, 71, 40, 40, 40, 40, 40]
itv_death = [40, 142, 40, 71, 40, 40, 40, 40, 40, 71, 71, 71, 40, 40, 71, 40, 40, 40, 40, 40, 71, 40, 71, 40, 40, 40, 71, 40, 40, 40, 40, 40, 40, 40, 71, 40, 40, 40, 71, 40, 40, 142, 40, 40, 71, 40, 71, 71, 40, 40]


offset_confirm = np.zeros(50)
offset_death = np.zeros(50)




res = []

def predict_inf (stateName, deg, date, offset):
    pd_state = df_total.loc[df_total['Province_State']==stateName]
    infected = pd_state['Confirmed'].to_numpy()[-date:]
    X = np.expand_dims(np.arange(0, len(infected)), axis=-1)
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, infected)

    X = np.expand_dims(np.arange(0, len(infected) + 26), axis=-1)
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X)
    predict = pol_reg.predict(X_poly) + offset
    return predict[-26:]

def predict_death (stateName, deg, date, offset):
    pd_state = df_total.loc[df_total['Province_State']==stateName]
    death = pd_state['Deaths'].to_numpy()[-date:]
    X = np.expand_dims(np.arange(0, len(death)), axis=-1)
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, death)

    X = np.expand_dims(np.arange(0, len(death) + 26), axis=-1)
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X)
    predict = pol_reg.predict(X_poly) + offset
    return predict[-26:]
    

for i in range (len(states)):
    res.append((predict_inf(states[i], deg_confirm[i], itv_confirm[i], offset_confirm[i]), predict_death(states[i], deg_death[i], itv_death[i], offset_death[i])))


# for i in range (len(states)):
#     for j in range (len(res[0][0])):

output = []
count = 0
for i in range (len(res[0][0])):
    for j in range (len(states)):
        output.append((count,res[j][0][i], res[j][1][i]))
        count += 1

output = pd.DataFrame(output, columns=['ForecastID', 'Confirmed', 'Deaths'])
output.to_csv("Team12.csv",index=False) 
