from parameters import *
from parameters import _lambda, _lambda_bar
from process import *
from process_plot import *
# from risk_plot import *
from scipy import interpolate
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt
import decimal
import concurrent.futures
import time
import os



def risk(model=hpp, T=50, tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0):
    np.random.seed()
    while True:
        if model == hpp:
            claim_time = hpp(T=T)
        elif model == npp:
            claim_time = npp(T=T)
        elif model == fhpp:
            claim_time = fhpp(T=T, alpha=alpha)
        elif model == fnpp:
            claim_time = fnpp(T=T, alpha=alpha)
        elif model == tfhpp:
            claim_time = tfhpp(tss_lambda=tss_1, T=T, alpha=alpha)
        elif model == mtfhpp:
            claim_time = mtfhpp(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, T=T, alpha=alpha, alpha_2=alpha_2)
        elif model == tfnpp:
            claim_time = tfnpp(tss_lambda=tss_1, T=T, alpha=alpha)
        elif model == mtfnpp:
            claim_time = mtfnpp(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, T=T, alpha=alpha, alpha_2=alpha_2)
        if len(claim_time) != 0:
                break
    number_of_claims = len(claim_time)
    claim_amount = claim(number_of_claims).tolist()
    claim_time_duplicate = [0]
    k = 0
    while k < number_of_claims:
        claim_time_duplicate.extend([claim_time[k]] * 2)
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
                claim_time_duplicate = claim_time_duplicate[0:k+2]
            else:
                claim_time_duplicate = claim_time_duplicate[0:k+1]
            return claim_time_duplicate, R_t
        elif k/2 == number_of_claims:
            return claim_time_duplicate, R_t
# risk()
def risk_alt(model=fhpp, T=50, tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2):
    mu = lambda_claim
    np.random.seed()
    while True:
        if model == fhpp:
            results = fhpp_ss(T=T, alpha=alpha)
        elif model == fnpp:
            results = fnpp_ss(T=T, alpha=alpha)
        elif model == tfhpp:
            results = tfhpp_ss(tss_lambda=tss_1, T=T, alpha=alpha)
        elif model == mtfhpp:
            results = mtfhpp_ss(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, T=T, alpha=alpha, alpha_2=alpha_2)
        elif model == tfnpp:
            results = tfnpp_ss(tss_lambda=tss_1, T=T, alpha=alpha)
        elif model == mtfnpp:
            results = mtfnpp_ss(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, T=T, alpha=alpha, alpha_2=alpha_2)
        if len(results[0]) != 0:
                break
    claim_time = results[0]
    iss_x, iss_y = results[2], results[1]
    f = interpolate.interp1d(iss_x, iss_y)
    number_of_claims = len(claim_time)
    claim_amount = claim(number_of_claims).tolist()
    time = []
    R_t = [u]
    k = 0
    while k < number_of_claims:
        time.append(claim_time[k])
        if k == 0:
            R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * f(claim_time[k]) - claim_amount[k])
        elif k > 0:
            R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * (f(claim_time[k]) - f(claim_time[k-1])) - claim_amount[k])
        k += 1
        if R_t[-1] < 0:
            return time, R_t
    time.append(T)
    # k = 0
    # j = 0
    # R_t = [u]
    return time, R_t
# risk_alt(T=10)

def risk_alt_plot(model=fhpp, T=50, tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2):
    mu = lambda_claim
    np.random.seed()
    while True:
        if model == fhpp:
            results = fhpp_ss(T=T, alpha=alpha)
        elif model == fnpp:
            results = fnpp_ss(T=T, alpha=alpha)
        elif model == tfhpp:
            results = tfhpp_ss(tss_lambda=tss_1, T=T, alpha=alpha)
        elif model == mtfhpp:
            results = mtfhpp_ss(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=c2, T=T)
        elif model == mtfnpp:
            results = mtfnpp_ss(tss_1=tss_1, tss_2=tss_2, c1=c1, c2=c2, T=T)
        if len(results[0]) != 0:
                break
    claim_time = results[0]
    iss_x, iss_y = results[2], results[1]
    f = interpolate.interp1d(iss_x, iss_y)
    number_of_claims = len(claim_time)
    claim_amount = claim(number_of_claims).tolist()
    time = [0]
    R_t = [u]
    k = 0
    while k < number_of_claims:
        time.extend([claim_time[k]
        , claim_time[k]
        ])
        if k == 0:
            R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * f(claim_time[k]))
            R_t.append(R_t[-1] - claim_amount[k])
            # R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * f(claim_time[k]) - claim_amount[k])
        elif k > 0:
            R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * f(claim_time[k]) - f(claim_time[k-1]))
            R_t.append(R_t[-1] - claim_amount[k])
            # R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * (f(claim_time[k]) - f(claim_time[k-1])) - claim_amount[k])
        k += 1
        if R_t[-1] < 0:
            return time, R_t
    time.append(T)
    R_t.append(R_t[-1] + mu * lambda_rate * (1+rho) * (f(T) - f(claim_time[k-1])))
    # k = 0
    # j = 0
    # R_t = [u]
    return time, R_t


def single_run_risk(model=hpp, T=50, tss_1=tss_1, tss_2=1 - tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0):
    if model == hpp or model == npp:
        if risk_model == 1:
            risk_model = 0
    if risk_model == 0:
        ruin = risk(model=model, T=T, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2)[0]
    elif risk_model == 1:
        risk_result = risk_alt(model=model, T=T, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2)
        ruin = risk_result[0]
    ruin_time = ruin[-1]
    if ruin[-1] != T:
        ruin = 1
    else:
        ruin = 0
    return ruin, ruin_time

def multiprocess(model=hpp, runs=1000, T=50, tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0):
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = [executor.submit(single_run_risk, model, T, tss_1, tss_2, c1, c2, alpha, alpha_2, risk_model) for _ in range(runs)]
        ruin = 0
        ruin_time = []
        for f in concurrent.futures.as_completed(results):
            ruin = ruin + f.result()[0]
            ruin_time.append(f.result()[1])
        ruin_prob = ruin/runs
    return ruin_prob, ruin_time

def multiprocess_itx(model=hpp, runs=1000, T=50, tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(single_run_risk, model, T, tss_1, tss_2, c1, c2, alpha, alpha_2, risk_model) for _ in range(runs)]
        ruin = 0
        ruin_time = []
        for f in concurrent.futures.as_completed(results):
            try:
                ruin = ruin + f.result()[0]
                ruin_time.append(f.result()[1])
            except Exception as e:
                print(f.result())
                print(e)
        ruin_prob = ruin/runs
    return ruin_prob, ruin_time

def multiprocess_siren(model=hpp, runs=1000, T=50, tss_1=tss_1, tss_2=1-tss_1, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0, core_count=siren_core_count):
    with concurrent.futures.ProcessPoolExecutor(max_workers=core_count) as executor:
        results = [executor.submit(single_run_risk, model, T, tss_1, tss_2, c1, c2, alpha, alpha_2, risk_model) for _ in range(runs)]
        ruin = 0
        ruin_time = []
        for f in concurrent.futures.as_completed(results):
            ruin = ruin + f.result()[0]
            ruin_time.append(f.result()[1])
        ruin_prob = ruin/runs
    return ruin_prob, ruin_time


