import time
import random #Default random number generator
import math
from scipy import stats

#sample from a Bernoulli(px/py) distribution
#px, py are integers
def sample_bernoulli(px,py,rng):
    assert 0 <= px/py <= 1
    m = rng.randrange(py)
    if m < px:
        return 1
    else:
        return 0

# sample from binomial(trials, px/py)
def BinomialInt(trials, px, py, rng):
    if trials < 0: return -1
    if trials == 0: return 0
    if px == 0: return 0
    if px == py: return trials
    r = 0
    for i in range(trials):
        if sample_bernoulli(px,py,rng) == 1:
            r = r + 1
    return r

# Algorithm 1 in Duchon and Duvignau, 2016.
def Poisson1(rng):
    n = 1
    g = 0
    k = 1
    while True:
        # generate a random integer from 1 to n+1, not including n+2
        i = rng.randrange(1,n+2)
        if i == n + 1:
            k = k + 1
        elif i > g:
            k = k - 1
            g = n + 1
        else:
            return k
        n = n + 1

def PoissonInt(mx, my, rng=None):
    if rng is None:
        rng = random.SystemRandom()
    # sample from Poisson with lamabda = mx/my, mx,my are integers
    if my == 0: return -1 # error
    if mx == 0 or (mx < 0 and my < 0) or (mx > 0 and my < 0): return 0
    r = 0
    while mx >= my:
        # deduce the parameter by 1
        r = r + Poisson1(rng)
        mx = mx - my
    if mx > 0:
        # see page 487 in Devroye, 1986.
        num = Poisson1(rng)
        r = r + BinomialInt(num, mx, my, rng)
    return r
