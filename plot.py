import numpy as np
from scipy import optimize
from numpy import absolute as abs
from numpy import pi, sin, cos, inf, log, exp
import matplotlib.pyplot as plt
import decimal
import concurrent.futures
import time
mu = 0

model = 'Weibull'
# model = 'Makeham'

# Parameters
alpha = 0.9
T = 50
# For homogeneous processes
lambda_rate = 3
# For nonhomogeneous processes
# limit_1 = 16
def lambda_model():
    if model == 'Makeham':
        return 0
    if model == 'Weibull':
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
u = 500 # initial capital
c = 20 # incoming capital per unit time

# Simulations
def hpp():
    # T_string = input("T:")
    # T = float(T_string)
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
        # Generate D ~ uniform(0,1);
        else:
            # Set n = n + 1
            n = n + 1

def npp():
    # T_string = input("T:")
    # T = float(T_string)
    # Initialize n = m = 0, t0 = s0 = 0, λ_bar = sup_{0≤t≤T} λ(t);
    m = 0
    n = 0
    t = [0]
    s = [0]
    lambda_bar = _lambda_bar()
    while s[m] < T:
        # Generate u ∼ uniform(0,1);
        # u = np.random.uniform()
        # Let w = - ln(u)/lambda_bar;
        # (_, neg_lambda_bar, _, _) = optimize.fminbound(lambda t: -_lambda(t), 0, T, full_output = 1)
        # lambda_bar = -neg_lambda_bar
        # t_lambda_bar = optimize.fminbound(lambda t: -_lambda(t), 0, T)
        # lambda_bar = _lambda(t_lambda_bar)
        w = np.random.exponential(1/lambda_bar)
        # w = - np.log(u)/lambda_bar
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
    if len(t) != 0:
        t.pop()
    # add T to the end of list
    # t.append(T)
    return t

def fhpp():
    # Fix the parameters λ > 0 and 0 < α < 1 for the FHPP
    _lambda = lambda_rate
    # Set n = 0 and t = 0.
    n = 0
    t = [0]
    while t[n] < T:
        # Generate three independent uniform random variables U_i ~ U(0, 1), i = 1, 2, 3.
        U = np.random.random_sample(3)
        dt = abs(log(U[0]))**(1 / alpha) / _lambda**(1 / alpha) * np.sin(alpha * np.pi * U[1]) * (np.sin(1 - alpha) * np.pi * U[1])**(1 / alpha - 1)/(np.sin(np.pi * U[1])**(1 / alpha) * abs(log(U[2]))**(1 / alpha - 1))
        t.append(t[n] + dt)
        n = n + 1
    if t[n] <= T:
        return print('error')
    # remove first 0 from list
    if len(t) != 0:
        t.pop()
    # remove last element larger than T from list
    t.pop()
    # add T to the end of list
    # t.append(T)
    return t

def fnpp():
    m = 0
    n = 0
    t = [0]
    s = [0]
    # t_lambda_bar = optimize.fminbound(lambda t: -_lambda(t), 0, T)
    lambda_bar = _lambda_bar()
    while s[m] < T:
        U = np.random.random_sample(3)
        dt = abs(log(U[0]))**(1 / alpha) / lambda_bar**(1 / alpha) * np.sin(alpha * np.pi * U[1]) * (np.sin(1 - alpha) * np.pi * U[1])**(1 / alpha - 1)/(np.sin(np.pi * U[1])**(1 / alpha) * abs(log(U[2]))**(1 / alpha - 1))
        # Set s(m+1) = s(m) + dt;
        s.append(s[m] + dt)
        # Generate U ~ uniform(0,1);
        U = np.random.uniform()
        if U <= (_lambda(s[m+1])/lambda_bar):
            # print(s)
            t.append(s[m+1])
            n = n + 1
        m = m + 1
    if t[n] <= T:
        t.pop(0)
        # t.append(T)
        return t
    else:
        # remove first 0 from list
        t.pop(0)
        # remove last element larger than T from list
        if len(t) != 0:
            t.pop()
        # add T to the end of list
        # t.append(T)
        return t


# claim_time = npp()
# n_bins = T
# _, ax = plt.subplots(figsize=(12, 9))
# _, _, _ = ax.hist(claim_time+[T], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
# plt.xlim(xmin = 0, xmax = T)
# plt.ylim(ymin = 0, ymax = 1.05)
# ax.set_title(r'NPP with $\lambda(t) = \frac{' + '{}'.format(c_lambda) + '}{3}$')
# plt.show()
# Plots of processes
def pp_plot(model = hpp, plot = 1):
    # plt.clf()
    if model == hpp:
        claim_time = hpp()
    elif model == npp:
        claim_time = npp()
    elif model == fhpp:
        claim_time = fhpp()
    elif model == fnpp:
        claim_time = fnpp()
    n_bins = T
    _, ax = plt.subplots(figsize=(12, 9))
    _, _, _ = ax.hist(claim_time+[T], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    plt.xlim(xmin = 0, xmax = T)
    plt.ylim(ymin = 0, ymax = 1.05)
    if model == hpp:
        ax.set_title(r'HPP with $\lambda = $' + '${}$'.format(lambda_rate))
    elif model == npp:
        if lambda_model() == 0:
            ax.set_title(r'NPP with $\lambda(t) = $' + '${}$'.format(c_lambda) + '$e^{' + format(b) + 't}$')
        elif lambda_model() == 1:
            # ax.set_title(r'NPP with $\lambda(t) = \frac{}'.format(c_lambda) + '{}$'.format(b) + '\frac{t}' + '{}$'.format(b))
            if c_lambda_b_mod == 0:
                ax.set_title(r'NPP with $\lambda(t) = ' + '{}$'.format(c_lambda_b) + r'$(' + '{}'.format(_1_b) + 't)^{' + '{}'.format(c_lambda_1) + '}$')
            elif c_lambda_b_mod != 0:
                ax.set_title(r'NPP with $\lambda(t) = \frac{' + '{}'.format(c_lambda) + '}{' + '{}'.format(b) + '}$' + r'$\left(\frac{t}{' + '{}'.format(b) + r'}\right)' + r'^{' + '{}'.format(c_lambda_1) + r'}$')
    elif model == fhpp:
        ax.set_title(r'FHPP with $\lambda = $' + '${}$'.format(lambda_rate) + r'$,\alpha = $' + '${}$'.format(alpha))
    elif model == fnpp:
        if lambda_model() == 0:
            ax.set_title(r'NPP with $\lambda(t) = $' + '${}$'.format(c_lambda) + '$e^{' + format(b) + 't}$' + r'$,\alpha = $' + '${}$'.format(alpha))
        elif lambda_model() == 1:
            if c_lambda_b_mod == 0:
                ax.set_title(r'FNPP with $\lambda(t) = ' + '{}$'.format(c_lambda_b) + r'$(' + '{}'.format(1/b) + 't)^{' + '{}'.format(c_lambda_1) + '}$' + r'$,\alpha = $' + '${}$'.format(alpha))
            elif c_lambda_b_mod != 0:
                ax.set_title(r'FNPP with $\lambda(t) = \frac{' + '{}'.format(c_lambda) + '}{' + '{}'.format(b) + '}$' + r'$\left(\frac{t}{' + '{}'.format(b) + r'}\right)' + r'^{' + '{}'.format(c_lambda_1) + r'}$' + r'$,\alpha = $' + '${}$'.format(alpha))
    if plot == 1:
        if model == hpp:
            plt.savefig('HPP.png')
        elif model == npp:
            plt.savefig('NPP.png')
        elif model == fhpp:
            plt.savefig('FHPP.png')
        elif model == fnpp:
            plt.savefig('FNPP.png')
    plt.show()

def hpp_plot():
    n_bins = T
    _, ax = plt.subplots(figsize=(8, 6))
    _, _, _ = ax.hist(hpp()+[T], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    ax.set_title(r'HPP with $\lambda = $' + '${}$'.format(lambda_rate))
    plt.xlim(xmin = 0, xmax = T)
    plt.ylim(ymin = 0, ymax = 1.05)
    plt.savefig('HPP.png')

def npp_plot():
    n_bins = T
    _, ax = plt.subplots(figsize=(8, 6))
    _, _, _ = ax.hist(npp()+[T], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    ax.set_title(r'NPP with $\lambda(t) = 3\sqrt{t}$ from $0\leq t<16$, $\frac{128}{t}$ from $16\leq t \leq 60$')
    plt.xlim(xmin = 0, xmax = T)
    plt.ylim(ymin = 0, ymax = 1.05)
    plt.savefig('NPP.png')

def fhpp_plot():
    n_bins = T
    _, ax = plt.subplots(figsize=(8, 6))
    _, _, _ = ax.hist(fhpp()+[T], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    ax.set_title(r'FHPP with $\lambda = $' + '${}$'.format(lambda_rate) + r'$,\alpha = $' + '${}$'.format(alpha))
    plt.xlim(xmin = 0, xmax = T)
    plt.ylim(ymin = 0, ymax = 1.05)
    plt.savefig('FHPP.png')

def fnpp_plot():
    n_bins = T
    _, ax = plt.subplots(figsize=(8, 6))
    _, _, _ = ax.hist(fnpp()+[T], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    ax.set_title(r'FNPP with $\lambda(t) = 3\sqrt{t}$ from $0\leq t<16$, $\frac{128}{t}$ from $16\leq t \leq 60$' + r'$,\alpha = $' + '${}$'.format(alpha))
    plt.xlim(xmin = 0, xmax = T)
    plt.ylim(ymin = 0, ymax = 1.05)
    plt.savefig('FNPP.png')

# Risk processes
def risk(model = hpp):
    while True:
        if model == hpp:
            claim_time = hpp()
        elif model == npp:
            claim_time = npp()
        elif model == fhpp:
            claim_time = fhpp()
        elif model == fnpp:
            claim_time = fnpp()
        if len(claim_time) != 0:
                break
    number_of_claims = len(claim_time)
    claim_amount = claim(number_of_claims).tolist()
    claim_time_duplicate = [0]
    k = 0
    while k < number_of_claims:
        claim_time_duplicate.append(claim_time[k])
        claim_time_duplicate.append(claim_time[k] + 0.0000000000000001)
        k += 1
    claim_time_duplicate.append(T)
    k = 0
    j = 0
    R_t = [u]
    while True:
        if k == 0:
            R_t.append(R_t[k] + c * claim_time[j])
            k += 1
            R_t.append(R_t[k] - claim_amount[j])
            j += 1
            k += 1
        elif j < number_of_claims:
            R_t.append(R_t[k] + c * (claim_time[j] - claim_time[j-1]))
            k += 1
            R_t.append(R_t[k] - claim_amount[j])
            j += 1
            k += 1
        if j == number_of_claims:
            R_t.append(R_t[k] + c * (T - claim_time[j-1]))
        if R_t[k] < 0:
            if len(claim_time_duplicate) == len(R_t):
                # print("ruined1")
                claim_time_duplicate = claim_time_duplicate[0:k+2]
            else:
                claim_time_duplicate = claim_time_duplicate[0:k+1]
                # print("ruined2")
            # return len(claim_time_duplicate), len(R_t)
            return claim_time_duplicate, R_t
        elif k/2 == number_of_claims:
            return claim_time_duplicate, R_t
            # return len(R_t), len(claim_time_duplicate)
            # return R_t, claim_time, claim_amount

def risk_plot(model = hpp, plot = 1):
    # plt.clf()
    plt.figure(figsize=(12,9))
    risk_x, risk_y = risk(model = model)
    plt.plot(risk_x, risk_y)
    plt.axhline(y = 0, c = 'black', linewidth = 0.75)
    plt.xlim(xmin = 0, xmax = min(T,max(risk_x))*1.05)
    plt.ylim(ymin = min(0,min(risk_y)), ymax = max(risk_y) + 10)
    # plt.show()
    if plot == 1:
        if model == hpp:
            plt.title("HPP")
            plt.savefig('HPP_risk.png')
        elif model == npp:
            plt.title("NPP")
            plt.savefig('NPP_risk.png')
        elif model == fhpp:
            plt.title("FHPP")
            plt.savefig('FHPP_risk.png')
        elif model == fnpp:
            plt.title("FNPP")
            plt.savefig('FNPP_risk.png')
    plt.show()
    # plt.savefig('HPP_risk.png')

def hpp_risk_plot():
    plt.clf()
    plt.figure(figsize=(12,9))
    risk_x, risk_y = risk(model = hpp)
    plt.plot(risk_x, risk_y)
    plt.axhline(y = 0, c = 'black', linewidth = 0.75)
    plt.title("HPP")
    plt.xlim(xmin = 0, xmax = min(T,max(risk_x))*1.05)
    plt.ylim(ymin = min(0,min(risk_y)), ymax = max(risk_y) + 10)
    # plt.show()
    plt.savefig('HPP_risk.png')

def npp_risk_plot():
    plt.clf()
    plt.figure(figsize=(12,9))
    risk_x, risk_y = risk(model = npp)
    plt.plot(risk_x, risk_y)
    plt.axhline(y = 0, c = 'black', linewidth = 0.75)
    plt.title("NPP")
    plt.xlim(xmin = 0, xmax = min(T,max(risk_x))*1.05)
    plt.ylim(ymin = min(0,min(risk_y)), ymax = max(risk_y) + 10)
    # plt.show()
    plt.savefig('NPP_risk.png')

def fhpp_risk_plot():
    plt.clf()
    plt.figure(figsize=(12,9))
    risk_x, risk_y = risk(model = fhpp)
    plt.plot(risk_x, risk_y)
    plt.axhline(y = 0, c = 'black', linewidth = 0.75)
    plt.title("FHPP")
    plt.xlim(xmin = 0, xmax = min(T,max(risk_x))*1.05)
    plt.ylim(ymin = min(0,min(risk_y)), ymax = max(risk_y) + 10)
    # plt.show()
    plt.savefig('FHPP_risk.png')

def fnpp_risk_plot():
    plt.clf()
    plt.figure(figsize=(12,9))
    risk_x, risk_y = risk(model = fnpp)
    plt.plot(risk_x, risk_y)
    plt.axhline(y = 0, c = 'black', linewidth = 0.75)
    plt.title("FNPP")
    plt.xlim(xmin = 0, xmax = min(T,max(risk_x))*1.05)
    plt.ylim(ymin = min(0,min(risk_y)), ymax = max(risk_y) + 10)
    # plt.show()
    plt.savefig('FNPP_risk.png')

def mult_run_risk(model = hpp, runs = 1000):
    ruin_time = []
    n = 0
    for f in range(runs):
        ruin = risk(model)[0]
        if ruin[-1] != T:
            n = n + 1
        ruin_time.append(ruin[-1])
        print(f+1)
    ruin_prob = n/runs
    return ruin_prob

def single_run_risk(model = hpp):
    np.random.seed()
    ruin = risk(model)[0]
    ruin_time = ruin[-1]
    if ruin[-1] != T:
        ruin = 1
    else:
        ruin = 0
    return ruin, ruin_time

def mult_run_risk_test(model = hpp, runs = 10000):
    start = time.perf_counter()
    ruin = 0
    for _ in range(runs):
        # print(f)
        ruin = ruin + single_run_risk(model)[0]
    finish = time.perf_counter()
    print('run time =', round(finish-start, 2))
    ruin_prob = ruin/runs
    return ruin_prob

def multiprocess(model = hpp, runs = 1000):
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # f1 = executor.submit(single_run_risk, npp)
        results = [executor.submit(single_run_risk, model) for _ in range(runs)]
        ruin = 0
        for f in concurrent.futures.as_completed(results):
            ruin = ruin + f.result()[0]
            # print(f)
        ruin_prob = ruin/runs
        # print(ruin_prob)
    finish = time.perf_counter()
    print('run time =', round(finish-start, 2), 'seconds')
    return ruin_prob
