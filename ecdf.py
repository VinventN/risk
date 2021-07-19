from parameters import *
from parameters import _lambda, _lambda_bar
from process import *
from process_plot import *
from risk_process import *
from risk_plot import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, norm

import csv
import pandas as pd
# from statsmodels.distributions.empirical_distribution import ECDF


fig = plt.figure(figsize=(8,3))
result = []
result_output_csv = []
def kde_single(model=hpp, runs=1000, T=1000, x_min=0, x_max=200, y_min=0, y_max=0.03, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0):
    # plt.figure(figsize=(12,9))
    sample = multiprocess(model=model, runs=runs, T=T, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=c2, alpha=alpha, alpha_2=alpha_2, risk_model=risk_model)[1]
    density = gaussian_kde(sample)
    xs = np.linspace(x_min, T, 5*T)
    plt.xlim(x_min, x_max)
    if model == hpp or model == npp:
        plot_ymax = round(max(density(xs)),2)+0.01
        # plt.ylim(y_min, plot_ymax)
    elif model == fhpp or model == fnpp:
        plot_ymax = round(max(density(xs)),3)+0.001
        # plt.ylim(y_min, plot_ymax)
    density.covariance_factor = lambda : 0.2
    density._compute_covariance()
    if model == hpp:
        plt.plot(xs,density(xs), label = 'HPP')
        model_type = 'hpp_model'
    elif model == npp:
        plt.plot(xs,density(xs), label = 'NPP')
        model_type = 'npp_model'
    elif model == fhpp:
        plt.plot(xs,density(xs), label = 'FHPP')
        model_type = 'fhpp_model'
    elif model == fnpp:
        plt.plot(xs,density(xs), label = 'FNPP')
        model_type = 'fnpp_model'
    elif model == tfhpp:
        plt.plot(xs,density(xs), label = 'TFHPP')
        model_type = 'tfhpp_model'
        plot_ymax = 0.007
    elif model == mtfhpp:
        plt.plot(xs,density(xs), label = 'MTFHPP')
        model_type = 'mtfhpp_model'
        plot_ymax = 0.007
    elif model == mtfnpp:
        plt.plot(xs,density(xs), label = 'MTFNPP')
        model_type = 'mtfnpp_model'
        plot_ymax = 0.006

    single_result = xs, density, x_max, plot_ymax, sample, model_type
    result_output_csv.append(single_result)
    # return xs, density, x_max, plot_ymax, sample, model_type

def kde_single_itx(model=hpp, runs=1000, T=1000, x_min=0, x_max=200, y_min=0, y_max=0.03, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0):
    # plt.figure(figsize=(12,9))
    sample = multiprocess_itx(model=model, runs=runs, T=T, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=c2, alpha=alpha, alpha_2=alpha_2, risk_model=risk_model)[1]
    density = gaussian_kde(sample)
    xs = np.linspace(x_min, T, 5*T)
    plt.xlim(x_min, x_max)
    plot_ymax = 0.008
    if model == hpp or model == npp:
        plot_ymax = round(max(density(xs)),2)+0.01
        # plt.ylim(y_min, plot_ymax)
    elif model == fhpp or model == fnpp:
        plot_ymax = round(max(density(xs)),3)+0.001
        # plt.ylim(y_min, plot_ymax)
    density.covariance_factor = lambda : 0.2
    density._compute_covariance()
    if model == hpp:
        plt.plot(xs,density(xs), label = 'HPP')
        model_type = 'hpp_model'
    elif model == npp:
        plt.plot(xs,density(xs), label = 'NPP')
        model_type = 'npp_model'
    elif model == fhpp:
        plt.plot(xs,density(xs), label = 'FHPP')
        model_type = 'fhpp_model'
    elif model == fnpp:
        plt.plot(xs,density(xs), label = 'FNPP')
        model_type = 'fnpp_model'
    elif model == tfhpp:
        plt.plot(xs,density(xs), label = 'TFHPP')
        model_type = 'tfhpp_model'
        plot_ymax = 0.006
    elif model == mtfhpp:
        plt.plot(xs,density(xs), label = 'MTFHPP')
        model_type = 'mtfhpp_model'
        plot_ymax = 0.007
    elif model == tfnpp:
        plt.plot(xs,density(xs), label = 'TFNPP')
        model_type = 'tfnpp_model'
        plot_ymax = 0.006
    elif model == mtfnpp:
        plt.plot(xs,density(xs), label = 'MTFNPP')
        model_type = 'mtfnpp_model'
        plot_ymax = 0.006
    # plt.plot(xs,density(xs))
    # if model == hpp:
    #     plt.title("HPP")
        # plt.savefig('HPP_kde.png')
    # elif model == npp:
    #     plt.title("NPP")
    #     plt.savefig('NPP_kde.png')
    # elif model == fhpp:
    #     plt.title("FHPP")
    #     plt.savefig('FHPP_kde.png')
    # elif model == fnpp:
    #     plt.title("FNPP")
    #     plt.savefig('FNPP_kde.png')
    # plt.savefig('ecdf.png')
    # plt.show()
    single_result = xs, density, x_max, plot_ymax, sample, model_type
    result_output_csv.append(single_result)
    # return xs, density, x_max, plot_ymax, sample, model_type

def kde_single_siren(model=hpp, runs=1000, T=1000, x_min=0, x_max=200, y_min=0, y_max=0.03, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=1-c1, alpha=alpha, alpha_2=alpha_2, risk_model=0, core_count=siren_core_count):
    sample = multiprocess_siren(model=model, runs=runs, T=T, tss_1=tss_1, tss_2=tss_2, c1=c1, c2=c2, alpha=alpha, alpha_2=alpha_2, risk_model=risk_model, core_count=core_count)[1]
    density = gaussian_kde(sample)
    xs = np.linspace(x_min, T, 5*T)
    plt.xlim(x_min, x_max)
    plot_ymax = 0.008
    if model == hpp or model == npp:
        plot_ymax = round(max(density(xs)),2)+0.01
    elif model == fhpp or model == fnpp:
        plot_ymax = round(max(density(xs)),3)+0.001
    density.covariance_factor = lambda : 0.2
    density._compute_covariance()
    if model == hpp:
        plt.plot(xs,density(xs), label = 'HPP')
        model_type = 'hpp_model'
    elif model == npp:
        plt.plot(xs,density(xs), label = 'NPP')
        model_type = 'npp_model'
    elif model == fhpp:
        plt.plot(xs,density(xs), label = 'FHPP')
        model_type = 'fhpp_model'
    elif model == fnpp:
        plt.plot(xs,density(xs), label = 'FNPP')
        model_type = 'fnpp_model'
    elif model == tfhpp:
        plt.plot(xs,density(xs), label = 'TFHPP')
        model_type = 'tfhpp_model'
        plot_ymax = 0.006
    elif model == mtfhpp:
        plt.plot(xs,density(xs), label = 'MTFHPP')
        model_type = 'mtfhpp_model'
        plot_ymax = 0.007
    elif model == tfnpp:
        plt.plot(xs,density(xs), label = 'TFNPP')
        model_type = 'tfnpp_model'
        plot_ymax = 0.006
    elif model == mtfnpp:
        plt.plot(xs,density(xs), label = 'MTFNPP')
        model_type = 'mtfnpp_model'
        plot_ymax = 0.006
    single_result = xs, density, x_max, plot_ymax, sample, model_type
    result_output_csv.append(single_result)

