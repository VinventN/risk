import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from numpy.random import normal
from numpy import hstack
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = hstack((sample1, sample2))
density = gaussian_kde(sample)
xs = np.linspace(0,80,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()

# plt.figure(figsize=(12,9))
sample = multiprocess(model, runs, T)[1]
density = gaussian_kde(sample)
xs = np.linspace(x_min, x_max, 200)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
density.covariance_factor = lambda : 0.2
density._compute_covariance()
plt.plot(xs,density(xs))
# if model == hpp:
#     plt.title("HPP")
#     plt.savefig('HPP_kde.png')
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
plt.show()