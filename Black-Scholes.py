import math
from scipy.stats import norm

def opt_price_BS(is_call, spot, strike, texp, vol, rd, rf):
    d1 = (math.log(spot / strike) + (rd - rf + 0.5 * vol ** 2) * texp) / (vol * math.sqrt(texp))
    d2 = d1 - vol * math.sqrt(texp)
    
    if is_call:
        price = spot * math.exp(-rf * texp) * norm.cdf(d1) - strike * math.exp(-rd * texp) * norm.cdf(d2)
    else:
        price = strike * math.exp(-rd * texp) * norm.cdf(-d2) - spot * math.exp(-rf * texp) * norm.cdf(-d1)
    
    return price