from parameters import *
from parameters import _lambda, _lambda_bar
from process import *
from process_plot import *
from risk_process import *

import numpy as np
import matplotlib.pyplot as plt

# Risk processes
def risk_plot(model=hpp, T=100, risk_model=0):
    plt.figure(figsize=(5,3))
    if risk_model == 0 or model == hpp or model == npp:
        risk_x, risk_y = risk(model=model, T=T)
    elif risk_model == 1:
        risk_x, risk_y = risk_alt_plot(model=model, T=T)
    plt.plot(risk_x, risk_y)
    plt.axhline(y=0, c='black', linewidth=0.75)
    plt.xlim(xmin=0, xmax=min(T,max(risk_x))*1.05)
    plt.ylim(ymin=min(0,min(risk_y)), ymax=max(risk_y)+10)
    plt.tight_layout()
    plt.show()

def risk_plot_test(model=hpp, T=100, risk_model=0):
    plt.figure(figsize=(5,3))
    while True:
        if risk_model == 0 or model == hpp or model == npp:
            risk_x, risk_y = risk(model=model, T=T)
        elif risk_model == 1:
            risk_x, risk_y = risk_alt_plot(model=model, T=T)
        if min(risk_y) < 0:
            break
    plt.plot(risk_x, risk_y)
    plt.axhline(y=0, c='black', linewidth=0.75)
    plt.xlim(xmin=0, xmax=min(T,max(risk_x))*1.05)
    plt.ylim(ymin=min(0,min(risk_y)), ymax=max(risk_y)+10)
    plt.tight_layout()
    plt.show()

# risk_plot(mtfhpp, risk_model=0)
# plt.savefig('mtfhpp_0')
# risk_plot(mtfhpp, risk_model=1)
# plt.savefig('mtfhpp_1')
# risk_plot(mtfhpp, risk_model=0)
# plt.savefig('mtfnpp_0')
# risk_plot(mtfnpp, risk_model=1)
# plt.savefig('mtfnpp_1')



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

def fhpp_risk_plot2():
    # plt.clf()
    plt.figure(figsize=(12,9))
    risk_x, risk_y = risk_alt(model = fhpp)
    plt.plot(risk_x, risk_y)
    plt.axhline(y = 0, c = 'black', linewidth = 0.75)
    plt.title("FHPP")
    plt.xlim(xmin = 0, xmax = min(T,max(risk_x))*1.05)
    plt.ylim(ymin = min(0,min(risk_y)), ymax = max(risk_y) + 10)
    plt.show()
    # plt.savefig('FHPP_risk.png')

# def mixfhpp_risk_plot(model=hpp, plot=1, T=50):
#     # plt.clf()
#     plt.figure(figsize=(5,3))
#     risk_x, risk_y = risk(model=model)
#     plt.plot(risk_x, risk_y)
#     plt.axhline(y=0, c='black', linewidth=0.75)
#     plt.xlim(xmin=0, xmax=min(T,max(risk_x))*1.05)
#     plt.ylim(ymin=min(0,min(risk_y)), ymax=max(risk_y)+10)
#     plt.show()