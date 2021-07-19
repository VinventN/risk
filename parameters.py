import numpy as np
from scipy import optimize
from numpy import absolute as abs
from numpy import pi, sin, cos, inf, log, exp
from math import floor, ceil
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in exp")

siren_core_count = 4


c1 = 0.75
c2 = 1 - c1
tss_1 = 0.3
tss_2 = 0.5

int_model = 'Weibull'
# int_model = 'Makeham'
# Parameters
alpha = 0.9
alpha_2 = 0.8
T = 3000
# For homogeneous processes
lambda_rate = 4
# For nonhomogeneous processes
# limit_1 = 16
# rho = 0.2

def lambda_model():
    if int_model == 'Makeham':
        return 0
    if int_model == 'Weibull':
        return 1

if lambda_model() == 0:
    c_lambda = 1.5
    b = 0.025
    mu = 0
elif lambda_model() == 1:
    c_lambda = 0.9
    b = 0.19
    c_lambda_b_mod = 1
    c_lambda_1 = round(c_lambda - 1,len(str(c_lambda).split('.')))
    c_lambda_b = c_lambda/b
    if len(str(c_lambda_b).split('.')[-1]) < 3 :
        c_lambda_b = round(c_lambda/b, len(str(c_lambda).split('.')) + 1)
        c_lambda_b_mod = 0
    _1_b = 1/b
    if _1_b % 1 == 0:
        _1_b = int(_1_b)

def _lambda(t):
    if lambda_model() == 0:
        return c_lambda * exp(b*t) + mu
    elif lambda_model() == 1:
        return c_lambda/b * (t/b) ** (c_lambda-1)
    # return 0.5 * t**0.5 *sin(t)
    # return c/b * (t/b)**(c-1)
    # if t < limit_1:
    #     return 3 * t ** 0.5
    # elif t >= limit_1:
    #     return 128 / t

def integral_lambda(x):
    if lambda_model() == 0:
        return c_lambda * (exp(b*x) - 1) + mu * x
    elif lambda_model() == 1:
        return (x/b)**c

def _lambda_bar():
    t_lambda_bar = optimize.fminbound(lambda t: -_lambda(t), 0, T)
    # t_lambda_bar_1 = optimize.fminbound(lambda t: -_lambda(t), 0, limit_1)
    # t_lambda_bar_2 = optimize.fminbound(lambda t: -_lambda(t), limit_1, T)
    lambda_bar = _lambda(t_lambda_bar)
    # lambda_bar = _lambda(max(t_lambda_bar_1, t_lambda_bar_2))
    return lambda_bar

# For compound Poisson processes
# Aggregate claims process
lambda_claim = 10 # expected claim size
def claim(t):
    claim = np.random.exponential(lambda_claim, t)
    return claim

# For risk processes
# u + ct
u = 500 # initial capital
c = 20 # incoming capital per unit time

# loading factor
rho = 0.1
