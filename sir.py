import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def sir_loss(params, infected, recovered, death, N, s_0, i_0, r_0):
    time = len(infected)
    beta, gamma, d = params

    def sir(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        D = y[3]
        return [-beta * S * I / N, beta * S * I / N - gamma * I - d * I, gamma * I, d * I]

    sol = solve_ivp(sir, [0, time], [s_0, i_0, r_0, 0], t_eval=np.arange(0, time, 1), vectorized=True)
    loss = np.linalg.norm(sol.y[1] - infected) + np.linalg.norm(sol.y[2] - recovered) + np.linalg.norm(sol.y[3] - death)
    return loss



class StateSir(object):
    def __init__(self, state_name, state_df, population, pred_time):
        self.name = state_name
        self.population = population
        self.infected = state_df['Confirmed'].to_numpy()
        self.recovered = state_df['Recovered'].fillna(0).to_numpy()
        self.death = state_df['Deaths'].to_numpy()
        self.i0 = self.infected[0]
        self.s0 = population - self.i0
        self.r0 = self.recovered[0]

        self.pred_time = pred_time

    def train_and_predict(self):
        sol = minimize(sir_loss, [0.1, 0.1, 0.1], args=(self.infected, self.recovered, self.death, self.population, self.s0, self.i0, self.r0),
                       method='L-BFGS-B', bounds=[(0, 0.4), (0, 0.4), (0, 0.4)])
        beta, gamma, d = sol.x

        gamma = 0.033
        print(beta)
        print(gamma)
        print(d)

        S = [self.s0]
        I = [self.i0]
        D = [0]
        N = self.population
        for t in range(self.pred_time):
            I.append(I[-1] + beta * S[-1] * I[-1] / N - gamma * I[-1] - d * I[-1])
            S.append(S[-1] - beta * S[-1] * I[-1] / N)
            D.append(D[-1] + d * I[-1])

        return I, D


def mape(pred, true):
    length = len(pred)
    res = 0
    for i in range(length):
        res += np.abs(pred[i] - true[i]) / true[i]
    return res / 142


if __name__ == "__main__":
    df_total = pd.read_csv('./ucla2020-cs145-covid19-prediction/train.csv')
    states = df_total['Province_State'].unique()
    state_model = StateSir('Arizona', df_total.loc[df_total['Province_State']=='Arizona'], 7290000, 140)
    pred_comfirmed, pred_death = state_model.train_and_predict()
    plt.plot(np.arange(0, len(pred_comfirmed)), pred_comfirmed)
    plt.plot(np.arange(0, 142), state_model.infected)
    plt.show()
    print(mape(pred_comfirmed, state_model.infected))
    print(mape(pred_death, state_model.death))



