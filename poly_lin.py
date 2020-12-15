from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_total = pd.read_csv('ucla2020-cs145-covid19-prediction/train.csv')

states = df_total['Province_State'].unique()


deg_confirm = np.ones(50)
deg_death = np.ones(50)

itv_confirm = np.ones(50)*5
itv_death = np.ones(50)*14

#speical treatments for states with nonlinear shape 
itv_confirm[10] = 40
itv_confirm[40] = 12
itv_confirm[48] = 40
itv_death[1] = 40
itv_death[11] = 60
itv_death[15] = 142
itv_death[33] = 40


offset_confirm = np.zeros(50)
offset_death = np.zeros(50)

offset_death[1] = 1
offset_death[10] = 10
offset_death[33] = 5

deg_death[15] = 3

res = []

def predict_inf (stateName, deg, date, offset):
    deg = int(deg)
    date = int(date)
    offset = int(offset)
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
    deg = int(deg)
    date = int(date)
    offset = int(offset)
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


output = []
count = 0
for i in range (len(res[0][0])):
    for j in range (len(states)):
        output.append((count,res[j][0][i], res[j][1][i]))
        count += 1

output = pd.DataFrame(output, columns=['ForecastID', 'Confirmed', 'Deaths'])
output.to_csv("Team12.csv",index=False) 



