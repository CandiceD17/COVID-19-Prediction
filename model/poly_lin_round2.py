from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_total = pd.read_csv('ucla2020-cs145-covid19-prediction/train_round2.csv')

states = df_total['Province_State'].unique()
death = pd.read_csv('ucla2020-cs145-covid19-prediction/death_from_1122.csv')
confirm = pd.read_csv('ucla2020-cs145-covid19-prediction/case_from_1122.csv')
# print (death.loc[death['States']=='Alabama'].to_numpy().T)
nonensemble_states_death = ['New Hampshire', 'Georgia', 'Vermont']
nonensemble_states_confirm = ['Wyoming','Georgia']

res = []

def poly_reg_inf(state, d, date):
    pd_state = df_total.loc[df_total['Province_State']==state] 
    infected = pd_state['Confirmed'].to_numpy()
    for i in range(2, 15): # append new dataset starting from 11.23 
        infected = np.append(infected, confirm.loc[confirm['States']==state].to_numpy().T[i])
    infected = infected[-date:]
    X = np.expand_dims(np.arange(0, len(infected)), axis=-1) 
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, infected)
    X = np.expand_dims(np.arange(0, len(infected) + 8), axis=-1) 
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(X)
    predict = pol_reg.predict(X_poly)
    return infected, predict

def poly_reg_death(state, d, date):
    pd_state = df_total.loc[df_total['Province_State']==state] 
    infected = pd_state['Deaths'].to_numpy()
    for i in range(2, 15): # append new dataset starting from 11.23 
        infected = np.append(infected, death.loc[death['States']==state].to_numpy().T[i])
    infected = infected[-date:]
    X = np.expand_dims(np.arange(0, len(infected)), axis=-1) 
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, infected)
    X = np.expand_dims(np.arange(0, len(infected) + 8), axis=-1) 
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(X)
    predict = pol_reg.predict(X_poly)
    return infected, predict

# interval = 14
# states = df_total['Province_State'].unique() 
# for i, state in enumerate(states):
    # if (state == 'Hawaii'):
    #     infected,predict_death = poly_reg_death(state,1,7)
        
    #     infected,_ = poly_reg_death(state,1,14)
    #     predict_death = predict_death[-7:]
    # elif(state == 'Vermont'):
    #     infected,predict_death_deg1 = poly_reg_death(state, 1, 14)
    #     _,predict_death_deg2 = poly_reg_death(state, 2, 40)
    #     # infected = (infected1 + infected2) / 2
    #     predict_death_deg1 = predict_death_deg1[-7:]
    #     predict_death_deg2 = predict_death_deg2[-7:]
    #     predict_death = (predict_death_deg1+predict_death_deg2)/2
    #     predict_death[-7:][0] = 79
    #     predict_death[-7:][1] = 79
    #     predict_death[-7:][2] = 79 # fix 12.6 - 12.8
    # elif(state == 'South Dakota'):
    #     infected,predict_death_deg1 = poly_reg_death(state, 1, 14)
    #     _,predict_death_deg2 = poly_reg_death(state, 1, 7)
    #     predict_death_deg1 = predict_death_deg1[-7:]
    #     predict_death_deg2 = predict_death_deg2[-7:]
    #     # infected = (infected1 + infected2)/2
    #     predict_death = (predict_death_deg1+predict_death_deg2)/2

### visualize confirm data ###
    # infected, _ = poly_reg_inf(state, 1, 14)
    # if state in nonensemble_states_confirm:
    #     _,predict_inf = poly_reg_inf(state, 1, 7)
    #     predict_inf = predict_inf[-7:]
    # else:
    #     _,predict_inf_deg1 = poly_reg_inf(state, 1, 7)
    #     _,predict_inf_deg2 = poly_reg_inf(state, 2, 40)
            
    #     predict_inf_deg1 = predict_inf_deg1[-7:]
    #     predict_inf_deg2 = predict_inf_deg2[-7:]
    #     predict_inf = (predict_inf_deg1+predict_inf_deg2)/2
    
    # plt.plot(np.arange(interval-1, interval+ 6), predict_inf) 
    # plt.plot(np.arange(0, interval), infected) 
    # plt.title(state)
    # plt.show()
### visualize death data ###
    # if state in nonensemble_states_death:
    #     infected,predict_death = poly_reg_death(state,1,7)
    #     infected,_ = poly_reg_death(state,1,14)
    #     predict_death = predict_death[-7:]
    # else:
    #     infected,_ = poly_reg_death(state,1,14)
    #     if state == 'Hawaii':
    #         _,predict_death_deg1 = poly_reg_death(state, 1, 7)
    #         _,predict_death_deg2 = poly_reg_death(state, 1, 5)
    #     else:
    #         _,predict_death_deg1 = poly_reg_death(state, 1, 7)
    #         _,predict_death_deg2 = poly_reg_death(state, 2, 40)
            
    #     predict_death_deg1 = predict_death_deg1[-7:]
    #     predict_death_deg2 = predict_death_deg2[-7:]
    #     # infected = (infected1 + infected2)/2
    #     predict_death = (predict_death_deg1+predict_death_deg2)/2
        
        
    # plt.plot(np.arange(interval-1, interval+ 6), predict_death) 
    # plt.plot(np.arange(0, interval), infected) 
    # plt.title(state)
    # plt.show()



for state in states:

    if state in nonensemble_states_confirm:
        _,predict_inf = poly_reg_inf(state, 1, 7)
        predict_inf = predict_inf[-7:]
    else:
        _,predict_inf_deg1 = poly_reg_inf(state, 1, 7)
        _,predict_inf_deg2 = poly_reg_inf(state, 2, 40)
            
        predict_inf_deg1 = predict_inf_deg1[-7:]
        predict_inf_deg2 = predict_inf_deg2[-7:]
        predict_inf = (predict_inf_deg1+predict_inf_deg2)/2

    if state in nonensemble_states_death:
        infected,predict_death = poly_reg_death(state,1,7)
        infected,_ = poly_reg_death(state,1,14)
        predict_death = predict_death[-7:]
    else:
        infected,_ = poly_reg_death(state,1,14)
        if state == 'Hawaii':
            _,predict_death_deg1 = poly_reg_death(state, 1, 7)
            _,predict_death_deg2 = poly_reg_death(state, 1, 5)
        else:
            _,predict_death_deg1 = poly_reg_death(state, 1, 7)
            _,predict_death_deg2 = poly_reg_death(state, 2, 40)
            
        predict_death_deg1 = predict_death_deg1[-7:]
        predict_death_deg2 = predict_death_deg2[-7:]
        # infected = (infected1 + infected2)/2
        predict_death = (predict_death_deg1+predict_death_deg2)/2

    res.append((predict_inf, predict_death))



output = []
count = 0
for i in range (len(res[0][0])):
    for j in range (len(states)):
        output.append((count,res[j][0][i], res[j][1][i]))
        count += 1

output = pd.DataFrame(output, columns=['ForecastID', 'Confirmed', 'Deaths'])
output.to_csv("Team12_round2.csv",index=False) 
