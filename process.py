from parameters import *
from parameters import _lambda, _lambda_bar

import numpy as np
from scipy import optimize
from numpy import absolute as abs
from numpy import pi, sin, cos, inf, log, exp
import matplotlib.pyplot as plt
# Simulations
def hpp(T=50):
    # Initialize n = 0, t0 = 0;
    n = 0
    t = []
    while True:
        # Generate w ∼ Exp(lambda);
        w = np.random.exponential(1/lambda_rate)
        if n == 0:
            t.append(w)
        elif n > 0:
            t.append(t[n-1] + w)
        if t[n] > T:
            t = t[:-1]
            # t.append(T)
            return t
        else:
            # Set n = n + 1
            n = n + 1

def npp(T=50):
    # Initialize n = m = 0, t0 = s0 = 0, λ_bar = sup_{0≤t≤T} λ(t);
    m = 0
    n = 0
    t = [0]
    s = [0]
    lambda_bar = _lambda_bar()
    while s[m] < T:
        # Generate w ∼ Exp(lambda_bar);
        w = np.random.exponential(1/lambda_bar)
        # Set s(m+1) = s(m) + w;
        s.append(s[m] + w)
        # Generate D ~ uniform(0,1);
        D = np.random.uniform()
        if D <= (_lambda(s[m+1])/lambda_bar):
            # print(s)
            t.append(s[m+1])
            n = n + 1
        m = m + 1
    # remove first 0 from list
    t.pop(0)
    # remove last element larger than T from list
    if len(t) >= 1:
        t.pop()
    return t

# Stable subordinator
def stable_subordinator():
    np.random.seed()
    U = np.random.random_sample(2)
    S = np.sin(alpha * np.pi * U[0]) * (np.sin((1 - alpha) * np.pi * U[0]))**(1 / alpha - 1)/(np.sin(np.pi * U[0])**(1 / alpha) * abs(log(U[1]))**(1 / alpha - 1))
    return S

# FHPP
def fhpp_ss(T=50, alpha=alpha):
    _lambda = lambda_rate
    n = 0
    t = [0]
    x = [0]
    y = [0]
    while t[n] < T:
        np.random.seed()
        ss = stable_subordinator()
        X = np.random.exponential(1 / _lambda)
        ss_time = abs(X)**(1 / alpha)
        S_t = ss * ss_time
        t.append(t[n] + S_t)
        n = n + 1
        x.extend([x[-1] + ss_time, x[-1] + ss_time])
        y.extend([y[-1], y[-1] + S_t])
    t.pop(0)
    if len(t) >= 1:
        t.pop()
        x = [n for n in x if n <= T]
        y = y[0:len(x)]
        x.append(T)
        y.append(y[-1])
        # print(x[-2],y[-2],x[-1],y[-1])
    return t, x, y

def fhpp(T=50, alpha=alpha):
    t, x, y = fhpp_ss(T=T, alpha=alpha)
    return t

# FNPP
def fnpp_ss(T=50, alpha=alpha):
    m = 0
    n = 0
    t = [0]
    s = [0]
    x = [0]
    y = [0]
    lambda_bar = _lambda_bar()
    while s[m] < T:
        np.random.seed()
        ss = stable_subordinator()
        X = np.random.exponential(1 / lambda_bar)
        ss_time = abs(X)**(1 / alpha)
        S_t = ss * ss_time
        s.append(s[-1] + S_t)
        x.extend([x[-1] + ss_time, x[-1] + ss_time])
        y.extend([y[-1], y[-1] + S_t])
        V = np.random.uniform()
        if V <= (_lambda(s[-1])/lambda_bar):
            t.append(s[-1])
            n = n + 1
        m = m + 1
    y_nonho = []
    for i in range(len(y)):
        y_nonho.append(integral_lambda(y[i]))
    y = y_nonho
    if len(t) != 0:
        x = [n for n in x if n <= T]
        y = y[0:len(x)]
        x.append(T)
        y.append(y[-1])
    t.pop(0)
    if t[-1] > T:
        if len(t) >= 1:
            t.pop()
    return t, x, y

def fnpp(T=50, alpha=alpha):
    t, x, y = fnpp_ss(T=T, alpha=alpha)
    return t
# fnpp()
# Tempered stable subordinator
def tss_accept(tss_lambda=tss_1):
    _lambda = lambda_rate
    while True:
        np.random.seed()
        U = np.random.uniform()
        X = np.random.exponential(1 / _lambda)
        ss = stable_subordinator()
        ss_time = abs(X)**(1 / alpha)
        S = ss * ss_time
        if U <= exp(-tss_lambda * S):
            return S

# Tempered FHPP
def tfhpp_ss(tss_lambda=tss_1, T=50, alpha=alpha):
    _lambda = lambda_rate
    n = 0
    t = [0]
    x = [0]
    y = [0]
    while t[n] < T:
        while True:
            np.random.seed()
            U = np.random.uniform()
            X = np.random.exponential(1 / _lambda)
            ss = stable_subordinator()
            ss_time = abs(X)**(1 / alpha)
            S = ss * ss_time
            if U <= exp(-tss_lambda * S):
                S_t = S
                break
        t.append(t[n] + S_t)
        n = n + 1
        x.extend([x[-1] + ss_time, x[-1] + ss_time])
        y.extend([y[-1], y[-1] + S_t])
    if len(t) >= 1:
        t.pop()
    t.pop()
    return t, x, y

def tfhpp(tss_lambda=tss_1, T=50, alpha=alpha):
    return tfhpp_ss(tss_lambda=tss_lambda, T=T, alpha=alpha)[0]

# Mixture Tempered FHPP
def mix_c1_ss(tss_lambda=tss_1, c1=c1, T=50, alpha=alpha):
    _lambda = lambda_rate
    n = 0
    t = [0]
    x = [0]
    y = []
    ss_list = []
    while t[n] < T:
        while True:
            np.random.seed()
            U = np.random.uniform()
            X = np.random.exponential(1 / _lambda)
            ss = stable_subordinator()
            ss_time = abs(X)**(1 / alpha)
            S = ss * ss_time
            if U <= exp(-tss_lambda * S):
                break
        S_t = S/c1
        t.append(t[n] + S_t)
        n = n + 1
        x.append(x[-1] + ss_time / c1)
        y.append(ss * c1)
        ss_list.append(ss)
    if len(t) != 0:
        t.pop(0)
        t.pop()
        x.pop(0)
    return t, x, y, ss_list

def mix_c1(tss_lambda=tss_1, c1=c1, T=50, alpha=alpha):
    return mix_c1_ss(tss_lambda=tss_lambda, c1=c1, T=T, alpha=alpha)[0]

def mix_c2_ss(tss_lambda=tss_2, c2=1-c1, T=50, alpha_2=alpha_2):
    _lambda = lambda_rate
    n = 0
    t = [0]
    x = [0]
    y = []
    ss_list = []
    time_list = []
    while t[n] < T:
        while True:
            np.random.seed()
            U = np.random.uniform()
            X = np.random.exponential(1 / _lambda)
            ss = stable_subordinator()
            ss_time = abs(X)**(1 / alpha)
            S = ss * ss_time
            if U <= exp(-tss_lambda * S):
                break
        S_t = S/c2
        t.append(t[n] + S_t)
        n = n + 1
        x.append(x[-1] + ss_time / c2)
        y.append(ss * c2)
        ss_list.append(ss)
        # time_list.append(ss_time)
    if len(t) >= 1:
        t.pop(0)
        t.pop()
        x.pop(0)
    return t, x, y, ss_list

def mix_c2(tss_lambda=tss_1, c2=1-c1, T=50, alpha_2=alpha_2):
    return mix_c2_ss(tss_lambda=tss_lambda, c2=c2, T=T, alpha=alpha)[0]

def mtfhpp_ss(tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, T=50, alpha=alpha, alpha_2=alpha_2):
    S_1, x1, y1, ss1 = mix_c1_ss(tss_lambda=tss_1, c1=c1, T=T, alpha=alpha)
    S_2, x2, y2, ss2 = mix_c2_ss(tss_lambda=tss_2, c2=c2, T=T, alpha_2=alpha_2)
    if len(S_2) >= 1:
        S_2.pop(0)
    S = S_1 + S_2
    S.sort()
    x = x1 + x2
    y = y1 + y2
    ss_list = ss1 + ss2
    zipped = zip(x, y, ss_list)
    zipped_sorted = sorted(zipped)
    zipped_list = list(zip(*zipped_sorted))
    x_sorted = list(zipped_list[0])
    y_sorted = list(zipped_list[1])
    ss_sorted = list(zipped_list[2])
    x_plot = [0]
    y_plot = [0]
    n = 0
    while True:
        x_plot.extend([x_plot[-1] + x_sorted[n], x_plot[-1] + x_sorted[n]])
        y_plot.extend([y_plot[-1], y_plot[-1] + y_sorted[n]])
        n += 1
        if y_plot[-1] > T:
            break
    return S, x_plot, y_plot

def mtfhpp(tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, T=50, alpha=alpha, alpha_2=alpha_2):
    S, x, y = mtfhpp_ss(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=c2, T=T, alpha=alpha, alpha_2=alpha_2)
    return S

# TFNPP
def tfnpp_ss(tss_lambda=tss_1, T=50, alpha=alpha):
    m = 0
    n = 0
    t = [0]
    s = [0]
    x = [0]
    y = [0]
    lambda_bar = _lambda_bar()
    while s[m] < T:
        while True:
            np.random.seed()
            U = np.random.uniform()
            X = np.random.exponential(1 / lambda_bar)
            ss = stable_subordinator()
            ss_time = abs(X)**(1 / alpha)
            S_t = ss * ss_time
            if U <= exp(-tss_lambda * S_t):
                break
        s.append(s[-1] + S_t)
        x.extend([x[-1] + ss_time, x[-1] + ss_time])
        y.extend([y[-1], y[-1] + S_t])
        V = np.random.uniform()
        if V <= (_lambda(s[-1])/lambda_bar):
            t.append(s[-1])
            n = n + 1
        m = m + 1
    y_nonho = []
    for i in range(len(y)):
        y_nonho.append(integral_lambda(y[i]))
    y = y_nonho
    if len(t) != 0:
        x = [n for n in x if n <= T]
        y = y[0:len(x)]
        x.append(T)
        y.append(y[-1])
    if t[n] > T:
        if len(t) != 0:
            t.pop()
    t.pop(0)
    return t, x, y

def tfnpp(tss_lambda=tss_1, T=50, alpha=alpha):
    t, x, y = tfnpp_ss(tss_lambda=tss_1, T=T, alpha=alpha)
    return t
# tfnpp_ss()
# Mixture Tempered FNPP
def mix_nc1_ss(tss_lambda=tss_1, c1=c1, T=50, alpha=alpha):
    m = 0
    n = 0
    t = [0]
    s = [0]
    x = [0]
    y = []
    ss_list = []
    lambda_bar = _lambda_bar()
    while s[m] < T:
        while True:
            np.random.seed()
            U = np.random.uniform()
            X = np.random.exponential(1 / lambda_bar)
            ss = stable_subordinator()
            ss_time = abs(X)**(1 / alpha)
            S = ss * ss_time
            if U <= exp(-tss_lambda * S):
                break
        S_t = S/c1
        s.append(s[-1] + S_t)
        x.append(x[-1] + ss_time / c1)
        y.append(ss * c1)
        ss_list.append(ss)
        V = np.random.uniform()
        if V <= (_lambda(s[-1])/lambda_bar):
            t.append(s[-1])
            n = n + 1
        m = m + 1
    x.pop(0)
    t.pop(0)
    if t[-1] > T:
        if len(t) != 0:
            t.pop()
    return t, x, y, ss_list

def mix_nc1(tss_lambda=tss_1, c1=c1, T=50, alpha=alpha):
    t, x, y, ss_list = mix_nc1_ss(tss_lambda=tss_1, c1=c1, T=T, alpha=alpha)
    return t

def mix_nc2_ss(tss_lambda=tss_2, c2=1-c1, T=50, alpha_2=alpha_2):
    m = 0
    n = 0
    t = [0]
    s = [0]
    x = [0]
    y = []
    ss_list = []
    lambda_bar = _lambda_bar()
    while s[m] < T:
        while True:
            np.random.seed()
            U = np.random.uniform()
            X = np.random.exponential(1 / lambda_bar)
            ss = stable_subordinator()
            ss_time = abs(X)**(1 / alpha)
            S = ss * ss_time
            if U <= exp(-tss_lambda * S):
                break
        S_t = S/c2
        s.append(s[-1] + S_t)
        x.append(x[-1] + ss_time / c2)
        y.append(ss * c2)
        ss_list.append(ss)
        V = np.random.uniform()
        if V <= (_lambda(s[-1])/lambda_bar):
            t.append(s[-1])
            n = n + 1
        m = m + 1
    x.pop(0)
    t.pop(0)
    if t[-1] > T:
        if len(t) >= 1:
            t.pop()
    return t, x, y, ss_list

def mix_nc2(tss_lambda=tss_2, c2=1-c1, T=50, alpha_2=alpha_2):
    t, x, y, ss_list = mix_nc2_ss(tss_lambda=tss_2, c2=c2, T=T, alpha_2=alpha_2)
    return t

def mtfnpp_ss(tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, T=50, alpha=alpha, alpha_2=alpha_2):
    S_1, x1, y1, ss1 = mix_c1_ss(tss_lambda=tss_1, c1=c1, T=T, alpha=alpha)
    S_2, x2, y2, ss2 = mix_c2_ss(tss_lambda=tss_2, c2=c2, T=T, alpha_2=alpha_2)
    if len(S_2) != 0:
        S_2.pop(0)
    S = S_1 + S_2
    S.sort()
    x = x1 + x2
    y = y1 + y2
    ss_list = ss1 + ss2
    zipped = zip(x, y, ss_list)
    zipped_sorted = sorted(zipped)
    zipped_list = list(zip(*zipped_sorted))
    x_sorted = list(zipped_list[0])
    y_sorted = list(zipped_list[1])
    ss_sorted = list(zipped_list[2])
    x_plot = [0]
    y_plot = [0]
    n = 0
    while True:
        x_plot.extend([x_plot[-1] + x_sorted[n], x_plot[-1] + x_sorted[n]])
        y_plot.extend([y_plot[-1], y_plot[-1] + y_sorted[n]])
        n += 1
        if y_plot[-1] > T:
            break
    return S, x_plot, y_plot

def mtfnpp(tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, T=50, alpha=alpha, alpha_2=alpha_2):
    S, x, y = mtfnpp_ss(tss_1, tss_2, c1, c2, T, alpha, alpha_2)
    return S

