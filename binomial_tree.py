import bisect
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

def opt_price_tree(is_call, spot, strikes, strike_times, texp, vol, rd, rf, n_steps_min=30, n_steps_max=1000):
    """Calculates an American option price on an asset following geometric Brownian motion
    with constant drift and constant volatility, using a binomial tree. Allows for an option strike price
    which varies deterministically with time.

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

    # figure out the number of time steps to use
    approx_dt = 1e-5 / vol / vol
    n_steps = min(n_steps_max, max(n_steps_min, int(texp / approx_dt)))
    dt = texp / n_steps

    # get the spot scale factors for move up and move down from time step to time step; these are
    # set to match the first and second moments of the distribution of change in spot.

    up_fact = math.exp((rd - rf) * dt) * (1 + math.sqrt(math.exp(vol * vol * dt) - 1))
    dn_fact = math.exp((rd - rf) * dt) * (2 - math.exp(vol * vol * dt)) / (1 + math.sqrt(math.exp(vol * vol * dt) - 1))

    # calculate the discount factor between time steps

    disc = math.exp(-rd * dt)

    steps = np.arange(n_steps + 1)
    up_steps = np.power(up_fact, steps)
    dn_steps = np.power(dn_fact, n_steps - steps)

    phi = 1 if is_call else -1

    strike_index = bisect.bisect_left(strike_times, texp)
    if strike_index == len(strike_times):
        strike_index -= 1

    strike = strikes[strike_index]
    vals = np.maximum(0, phi * (spot * up_steps * dn_steps - strike))
    for i in range(n_steps - 1, -1, -1):
        # check whether the strike has changed; need to be careful about numerical precision here
        if strike_index > 0 and i * dt - strike_times[strike_index - 1] < 1e-12:
            strike_index -= 1
            strike = strikes[strike_index]

        up_steps = up_steps[:-1]
        dn_steps = dn_steps[1:]
        vals = np.maximum(phi * (spot * up_steps * dn_steps - strike), disc * 0.5 * (vals[1:] + vals[:-1]))

    price = vals[0]
    return price


def opt_price_CRR_tree(is_call, spot, strikes, strike_times, texp, vol, rd, rf, n_steps_min=30, n_steps_max=1000):

    if vol <= 0 or texp <= 0:
        return opt_price_BS(is_call, spot, strikes[-1], texp, vol, rd, rf)

    # figure out the number of time steps to use
    approx_dt = 1e-5 / vol / vol
    n_steps = min(n_steps_max, max(n_steps_min, int(texp / approx_dt)))
    dt = texp / n_steps

    # get the spot scale factors for move up and move down from time step to time step; these are
    # set to match the first and second moments of the distribution of change in spot.
    c = 0.5 * (math.exp(-(rd - rf) * dt) + math.exp((rd - rf + vol ** 2) * dt))
    dn_fact = c - math.sqrt(c ** 2 - 1)
    up_fact = 1 / dn_fact

    up_prob = (math.exp((rd - rf) * dt) - dn_fact) / (up_fact - dn_fact)
    dn_prob = 1 - up_prob

    # calculate the discount factor between time steps

    disc = math.exp(-rd * dt)

    steps = np.arange(n_steps + 1)
    up_steps = np.power(up_fact, steps)
    dn_steps = np.power(dn_fact, n_steps - steps)

    phi = 1 if is_call else -1

    strike_index = bisect.bisect_left(strike_times, texp)
    if strike_index == len(strike_times):
        strike_index -= 1

    strike = strikes[strike_index]
    vals = np.maximum(0, phi * (spot * up_steps * dn_steps - strike))
    for i in range(n_steps - 1, -1, -1):
        # check whether the strike has changed; need to be careful about numerical precision here
        if strike_index > 0 and i * dt - strike_times[strike_index - 1] < 1e-12:
            strike_index -= 1
            strike = strikes[strike_index]

        up_steps = up_steps[:-1]
        dn_steps = dn_steps[1:]
        vals = np.maximum(phi * (spot * up_steps * dn_steps - strike), disc * (up_prob * vals[1:] + dn_prob * vals[:-1]))

    price = vals[0]
    return price