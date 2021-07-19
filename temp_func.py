from ecdf import *
# from open_save import *
from parameters import *
from parameters import _lambda, _lambda_bar
from process import *
from process_plot import *
from risk_process import *
from risk_plot import *
import os
def save_csv(name='file.csv'):
    df1 = result_output_csv[0][4]
    df = pd.DataFrame(df1)
    df.to_csv(name, index=False, header=False)

def atx(runs=20000):
    kde_single(model=tfhpp, runs=runs, risk_model=0, T=T)
    save_csv('tfhpp_0_0.9_0.3_new.csv')
    print(min(result_output_csv[0][4]))

def itx(runs=1000000):
    kde_single_itx(model=fnpp, runs=runs, risk_model=0, T=2000)
    save_csv('fnpp_0_0.9_T_2000_aws.csv')
    print(min(result_output_csv[0][4]))

def siren(runs=200000, core_count=4):
    kde_single_siren(model=fnpp, runs=runs, risk_model=1, core_count=core_count, T=T)
    save_csv('fnpp_1_0.9_500_siren.csv')
    print(min(result_output_csv[0][4]))

itx()