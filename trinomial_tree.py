import math
import numpy as np
from scipy.stats import norm

def opt_price_BS(is_call, spot, strike, texp, vol, rd, rf):
    d1 = (math.log(spot / strike) + (rd - rf + 0.5 * vol ** 2) * texp) / (vol * math.sqrt(texp))
    d2 = d1 - vol * math.sqrt(texp)
    
    if is_call:
        price = spot * math.exp(-rf * texp) * norm.cdf(d1) - strike * math.exp(-rd * texp) * norm.cdf(d2)
    else:
        price = strike * math.exp(-rd * texp) * norm.cdf(-d2) - spot * math.exp(-rf * texp) * norm.cdf(-d1)
    
    return price

def opt_price_tri_tree(is_call, spot, strikes, strike_times, texp, vol, rd, rf, n_steps_min=30, n_steps_max=1000):
    """Calculates an American option price on an asset following geometric Brownian motion
    with constant drift and constant volatility, using a equal-probability trinomial tree. Allows for an 
    option strike price which varies deterministically with time.

    is_call: True, call option; False, put option
    spot: initial spot asset price
    strikes: time-dependent strike price of the option
    strike_times: break times for strike prices
    texp: time to expiration of the option
    vol: volatility of the option
    rd: discount rate
    rf: asset yield
    n_steps_min: minimum number of time steps to use
    n_steps_max: maximum number of time steps to use"""

    if vol <= 0 or texp <= 0:
        return opt_price_BS(is_call, spot, strikes[-1], texp, vol, rd, rf)

    dt_approx = 1e-5 / vol / vol
    n_steps = int(min(n_steps_max, max(n_steps_min, texp / dt_approx)))
    dt = texp / n_steps

    M = math.exp((rd - rf) * dt)
    V = math.exp(vol * vol * dt)
    K = M * (V + 3) / 4
    md_fact = M * (3 - V) / 2
    up_fact = K + math.sqrt(K * K - md_fact * md_fact)
    dn_fact = K - math.sqrt(K * K - md_fact * md_fact)

    up_prob = 1 / 3
    dn_prob = 1 / 3
    md_prob = 1 / 3

    disc = np.exp(-rd * dt)

    stock_prices = [np.array([spot])]

    for i in range(n_steps):
        prev_nodes = stock_prices[-1]
        up_nodes = prev_nodes * up_fact
        md_nodes = prev_nodes * md_fact
        dn_nodes = prev_nodes * dn_fact

        nodes = np.concatenate((up_nodes, [md_nodes[-1], dn_nodes[-1]]))
        stock_prices.append(nodes)

    phi = 1 if is_call else -1

    strike_index = np.searchsorted(strike_times, texp, side='right') - 1
    strike = strikes[strike_index]
    option_values = np.maximum(phi * (stock_prices[-1] - strike), 0)

    for i in range(n_steps - 1, -1, -1):
        next_values = option_values
        option_values = disc * (up_prob * next_values[:-2] + md_prob * next_values[1:-1] + dn_prob * next_values[2:])
        if strike_index > 0 and i * dt <= strike_times[strike_index - 1]:
            strike_index -= 1
            strike = strikes[strike_index]
        option_values = np.maximum(phi * (stock_prices[i] - strike), option_values)

    return option_values[0]