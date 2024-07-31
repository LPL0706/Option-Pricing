import math
import scipy.optimize
from scipy.stats import norm

def opt_price_BS(is_call, spot, strike, texp, vol, rd, rf):
    d1 = (math.log(spot / strike) + (rd - rf + 0.5 * vol ** 2) * texp) / (vol * math.sqrt(texp))
    d2 = d1 - vol * math.sqrt(texp)
    
    if is_call:
        price = spot * math.exp(-rf * texp) * norm.cdf(d1) - strike * math.exp(-rd * texp) * norm.cdf(d2)
    else:
        price = strike * math.exp(-rd * texp) * norm.cdf(-d2) - spot * math.exp(-rf * texp) * norm.cdf(-d1)
    
    return price

def opt_price_BAW(is_call, spot, strike, texp, vol, rd, rf, linear):
    """
    is_call: True, call option; False, put option
    spot: initial spot asset price
    strike: strike price of the option
    texp: time to expiration of the option
    vol: volatility of the option
    rd: discount rate
    rf: asset yield"""
    
    if rd == 0:
        rd = 1e-06
        
    if strike <= 0:
        if not is_call:
            return 0
        return max(math.exp(-rf * texp) * spot - math.exp(-rd * texp) * strike, spot - strike)

    euro_price = opt_price_BS(is_call, spot, strike, texp, vol, rd, rf)
    if vol <= 0 or texp <= 0:
        return euro_price

    phi = 1 if is_call else -1

    if rd == 0:
        h = 0
        alpha = 0
        ratio = 2. / vol / vol / texp
    else:
        h = -math.expm1(-rd * texp)
        alpha = 2 * rd / vol / vol
        ratio = alpha / h

    ratio = 2. / vol / vol / texp

    beta = 2 * (rd - rf) / vol / vol

    lam_arg = math.sqrt((beta - 1) * (beta - 1) + 4. * ratio)

    lam = 0.5 * (-(beta - 1) + phi * lam_arg)
    exprf = math.exp(-rf * texp)
    sqrtT = math.sqrt(texp)

    def arg_func(spotC):
        d1 = (math.log(spotC / strike) + (rd - rf + vol * vol / 2.) * texp) / vol / sqrtT
        N = norm.cdf(phi * d1)
        euro = bs_opt_price(is_call, spotC, strike, texp, vol, rd, rf)
        return phi * (exprf * N - 1) + lam * (phi * (spotC - strike) - euro) / spotC

    if is_call:
        lo_spot = strike
        hi_spot = strike * math.exp(6 * vol * sqrtT)
        
        # if the arg func is negative at the high strike, the barrier is too far away to hit
        # and the option is worth the European value

        if arg_func(hi_spot) < 0:
            return euro_price
    else:
        lo_spot = strike * math.exp(-6 * vol * sqrtT)
        hi_spot = strike
        
        # if the arg func is positive at the low strike, the barrier is too far away to hit
        # and the option is worth the European value

        if arg_func(lo_spot) > 0:
            return euro_price

    spotC = scipy.optimize.brenth(arg_func, lo_spot, hi_spot)
    
    # if spot's past the barrier, return intrinsic value

    if phi * (spotC - spot) <= 0:
        return phi * (spot - strike)

    return euro_price + hA * math.pow(spot / spotC, lam)