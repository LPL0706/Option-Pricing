import numpy as np

def regression(X, y, k, type):
    if type == 'laguerre':
        poly = np.polynomial.laguerre.lagval(X, np.eye(k + 1))
    elif type == 'hermite':
        poly = np.polynomial.hermite.hermval(X, np.eye(k))
    elif type == 'simple':
        poly = np.polynomial.polynomial.polyvander(X, k).T
    
    poly = np.column_stack(poly)
    coefficients = np.linalg.lstsq(poly, y, rcond=None)[0]
    return poly @ coefficients

def LSMC(S0, K, T, r, sigma, N, M, k, type='laguerre'):
    dt = T / M
    paths = int(N / 2)
    
    S = np.zeros((N, M + 1))
    S[:, 0] = S0
    for j in range(1, M + 1):
        Z = np.random.normal(0, 1, paths)
        Z = np.concatenate((Z, -Z))
        S[:, j] = S[:, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
    payoff = np.maximum(K-S,0)
    ind = np.zeros((N, M + 1))
    ind[S[:,-1] > K,-1] = 1
    option_price = np.copy(payoff)
    
    for i in range(M-1, 0, -1):
        in_the_money = S[:, i] < K
        X = S[in_the_money, i]
        y = np.zeros(N)
        for j in range(i+1,M):
            y += ind[:,j] * payoff[:,j] * np.exp(-r*dt*(i-j))
        Y = y[in_the_money]
        continuation_value = np.copy(payoff[:,i])
        continuation_value[in_the_money]= regression(X, Y, k, type)
        immediate_exercise_value = np.maximum(K - S[:, i], 0)
        for j in range(N):
            if i == M-1:
                break
            if immediate_exercise_value[j]>continuation_value[j]:
                ind[j,i] = 1
                ind[j,i+1:] = np.zeros(M-i)

        option_price[:,i] = np.where(immediate_exercise_value > continuation_value, immediate_exercise_value, payoff[:,i])
    
    return np.sum(option_price*ind)/N